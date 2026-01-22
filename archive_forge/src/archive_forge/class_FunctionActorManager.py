import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
class FunctionActorManager:
    """A class used to export/load remote functions and actors.
    Attributes:
        _worker: The associated worker that this manager related.
        _functions_to_export: The remote functions to export when
            the worker gets connected.
        _actors_to_export: The actors to export when the worker gets
            connected.
        _function_execution_info: The function_id
            and execution_info.
        _num_task_executions: The function
            execution times.
        imported_actor_classes: The set of actor classes keys (format:
            ActorClass:function_id) that are already in GCS.
    """

    def __init__(self, worker):
        self._worker = worker
        self._functions_to_export = []
        self._actors_to_export = []
        self._function_execution_info = defaultdict(lambda: {})
        self._num_task_executions = defaultdict(lambda: {})
        self.imported_actor_classes = set()
        self._loaded_actor_classes = {}
        self.lock = threading.RLock()
        self.cv = threading.Condition(lock=self.lock)
        self.execution_infos = {}
        self._num_exported = 0
        self._export_lock = threading.Lock()

    def increase_task_counter(self, function_descriptor):
        function_id = function_descriptor.function_id
        self._num_task_executions[function_id] += 1

    def get_task_counter(self, function_descriptor):
        function_id = function_descriptor.function_id
        return self._num_task_executions[function_id]

    def compute_collision_identifier(self, function_or_class):
        """The identifier is used to detect excessive duplicate exports.
        The identifier is used to determine when the same function or class is
        exported many times. This can yield false positives.
        Args:
            function_or_class: The function or class to compute an identifier
                for.
        Returns:
            The identifier. Note that different functions or classes can give
                rise to same identifier. However, the same function should
                hopefully always give rise to the same identifier. TODO(rkn):
                verify if this is actually the case. Note that if the
                identifier is incorrect in any way, then we may give warnings
                unnecessarily or fail to give warnings, but the application's
                behavior won't change.
        """
        import io
        string_file = io.StringIO()
        dis.dis(function_or_class, file=string_file, depth=2)
        collision_identifier = function_or_class.__name__ + ':' + string_file.getvalue()
        return hashlib.sha1(collision_identifier.encode('utf-8')).digest()

    def load_function_or_class_from_local(self, module_name, function_or_class_name):
        """Try to load a function or class in the module from local."""
        module = importlib.import_module(module_name)
        parts = [part for part in function_or_class_name.split('.') if part]
        object = module
        try:
            for part in parts:
                object = getattr(object, part)
            return object
        except Exception:
            return None

    def export_key(self, key):
        """Export a key so it can be imported by other workers"""
        with self._export_lock:
            while True:
                self._num_exported += 1
                holder = make_export_key(self._num_exported, self._worker.current_job_id)
                if self._worker.gcs_client.internal_kv_put(holder, key, False, KV_NAMESPACE_FUNCTION_TABLE) > 0:
                    break
        self._worker.gcs_publisher.publish_function_key(key)

    def export_setup_func(self, setup_func: Callable, timeout: Optional[int]=None) -> bytes:
        """Export the setup hook function and return the key."""
        pickled_function = pickle_dumps(setup_func, f'Cannot serialize the worker_process_setup_hook {setup_func.__name__}')
        function_to_run_id = hashlib.shake_128(pickled_function).digest(ray_constants.ID_SIZE)
        key = make_function_table_key(WORKER_PROCESS_SETUP_HOOK_KEY_NAME_GCS.encode(), self._worker.current_job_id.binary(), function_to_run_id)
        check_oversized_function(pickled_function, setup_func.__name__, 'function', self._worker)
        try:
            self._worker.gcs_client.internal_kv_put(key, pickle.dumps({'job_id': self._worker.current_job_id.binary(), 'function_id': function_to_run_id, 'function': pickled_function}), True, ray_constants.KV_NAMESPACE_FUNCTION_TABLE, timeout=timeout)
        except Exception as e:
            logger.exception(f'Failed to export the setup hook {setup_func.__name__}.')
            raise e
        return key

    def export(self, remote_function):
        """Pickle a remote function and export it to redis.
        Args:
            remote_function: the RemoteFunction object.
        """
        if self._worker.load_code_from_local:
            function_descriptor = remote_function._function_descriptor
            module_name, function_name = (function_descriptor.module_name, function_descriptor.function_name)
            if self.load_function_or_class_from_local(module_name, function_name) is not None:
                return
        function = remote_function._function
        pickled_function = remote_function._pickled_function
        check_oversized_function(pickled_function, remote_function._function_name, 'remote function', self._worker)
        key = make_function_table_key(b'RemoteFunction', self._worker.current_job_id, remote_function._function_descriptor.function_id.binary())
        if self._worker.gcs_client.internal_kv_exists(key, KV_NAMESPACE_FUNCTION_TABLE):
            return
        val = pickle.dumps({'job_id': self._worker.current_job_id.binary(), 'function_id': remote_function._function_descriptor.function_id.binary(), 'function_name': remote_function._function_name, 'module': function.__module__, 'function': pickled_function, 'collision_identifier': self.compute_collision_identifier(function), 'max_calls': remote_function._max_calls})
        self._worker.gcs_client.internal_kv_put(key, val, True, KV_NAMESPACE_FUNCTION_TABLE)

    def fetch_registered_method(self, key: str, timeout: Optional[int]=None) -> Optional[ImportedFunctionInfo]:
        vals = self._worker.gcs_client.internal_kv_get(key, KV_NAMESPACE_FUNCTION_TABLE, timeout=timeout)
        if vals is None:
            return None
        else:
            vals = pickle.loads(vals)
            fields = ['job_id', 'function_id', 'function_name', 'function', 'module', 'max_calls']
            return ImportedFunctionInfo._make((vals.get(field) for field in fields))

    def fetch_and_register_remote_function(self, key):
        """Import a remote function."""
        remote_function_info = self.fetch_registered_method(key)
        if not remote_function_info:
            return False
        job_id_str, function_id_str, function_name, serialized_function, module, max_calls = remote_function_info
        function_id = ray.FunctionID(function_id_str)
        job_id = ray.JobID(job_id_str)
        max_calls = int(max_calls)
        with self.lock:
            self._num_task_executions[function_id] = 0
            try:
                function = pickle.loads(serialized_function)
            except Exception:
                traceback_str = format_error_message(traceback.format_exc())

                def f(*args, **kwargs):
                    raise RuntimeError('The remote function failed to import on the worker. This may be because needed library dependencies are not installed in the worker environment:\n\n{}'.format(traceback_str))
                self._function_execution_info[function_id] = FunctionExecutionInfo(function=f, function_name=function_name, max_calls=max_calls)
                logger.debug(f"Failed to unpickle the remote function '{function_name}' with function ID {function_id.hex()}. Job ID:{job_id}.Traceback:\n{traceback_str}. ")
            else:
                function.__module__ = module
                self._function_execution_info[function_id] = FunctionExecutionInfo(function=function, function_name=function_name, max_calls=max_calls)
        return True

    def get_execution_info(self, job_id, function_descriptor):
        """Get the FunctionExecutionInfo of a remote function.
        Args:
            job_id: ID of the job that the function belongs to.
            function_descriptor: The FunctionDescriptor of the function to get.
        Returns:
            A FunctionExecutionInfo object.
        """
        function_id = function_descriptor.function_id
        if function_id in self._function_execution_info:
            return self._function_execution_info[function_id]
        if self._worker.load_code_from_local:
            if not function_descriptor.is_actor_method():
                if self._load_function_from_local(function_descriptor) is True:
                    return self._function_execution_info[function_id]
        with profiling.profile('wait_for_function'):
            self._wait_for_function(function_descriptor, job_id)
        try:
            function_id = function_descriptor.function_id
            info = self._function_execution_info[function_id]
        except KeyError as e:
            message = 'Error occurs in get_execution_info: job_id: %s, function_descriptor: %s. Message: %s' % (job_id, function_descriptor, e)
            raise KeyError(message)
        return info

    def _load_function_from_local(self, function_descriptor):
        assert not function_descriptor.is_actor_method()
        function_id = function_descriptor.function_id
        module_name, function_name = (function_descriptor.module_name, function_descriptor.function_name)
        object = self.load_function_or_class_from_local(module_name, function_name)
        if object is not None:
            function = object._function
            self._function_execution_info[function_id] = FunctionExecutionInfo(function=function, function_name=function_name, max_calls=0)
            self._num_task_executions[function_id] = 0
            return True
        else:
            return False

    def _wait_for_function(self, function_descriptor, job_id: str, timeout=10):
        """Wait until the function to be executed is present on this worker.
        This method will simply loop until the import thread has imported the
        relevant function. If we spend too long in this loop, that may indicate
        a problem somewhere and we will push an error message to the user.
        If this worker is an actor, then this will wait until the actor has
        been defined.
        Args:
            function_descriptor : The FunctionDescriptor of the function that
                we want to execute.
            job_id: The ID of the job to push the error message to
                if this times out.
        """
        start_time = time.time()
        warning_sent = False
        while True:
            with self.lock:
                if self._worker.actor_id.is_nil():
                    if function_descriptor.function_id in self._function_execution_info:
                        break
                    else:
                        key = make_function_table_key(b'RemoteFunction', job_id, function_descriptor.function_id.binary())
                        if self.fetch_and_register_remote_function(key) is True:
                            break
                else:
                    assert not self._worker.actor_id.is_nil()
                    assert self._worker.actor_id in self._worker.actors
                    break
            if time.time() - start_time > timeout:
                warning_message = f'This worker was asked to execute a function that has not been registered ({function_descriptor}, node={self._worker.node_ip_address}, worker_id={self._worker.worker_id.hex()}, pid={os.getpid()}). You may have to restart Ray.'
                if not warning_sent:
                    logger.error(warning_message)
                    ray._private.utils.push_error_to_driver(self._worker, ray_constants.WAIT_FOR_FUNCTION_PUSH_ERROR, warning_message, job_id=job_id)
                warning_sent = True
            self._worker.import_thread._do_importing()
            time.sleep(0.001)

    def export_actor_class(self, Class, actor_creation_function_descriptor, actor_method_names):
        if self._worker.load_code_from_local:
            module_name, class_name = (actor_creation_function_descriptor.module_name, actor_creation_function_descriptor.class_name)
            if self.load_function_or_class_from_local(module_name, class_name) is not None:
                return
        assert not self._worker.current_job_id.is_nil(), 'You might have started a background thread in a non-actor task, please make sure the thread finishes before the task finishes.'
        job_id = self._worker.current_job_id
        key = make_function_table_key(b'ActorClass', job_id, actor_creation_function_descriptor.function_id.binary())
        serialized_actor_class = pickle_dumps(Class, f'Could not serialize the actor class {actor_creation_function_descriptor.repr}')
        actor_class_info = {'class_name': actor_creation_function_descriptor.class_name.split('.')[-1], 'module': actor_creation_function_descriptor.module_name, 'class': serialized_actor_class, 'job_id': job_id.binary(), 'collision_identifier': self.compute_collision_identifier(Class), 'actor_method_names': json.dumps(list(actor_method_names))}
        check_oversized_function(actor_class_info['class'], actor_class_info['class_name'], 'actor', self._worker)
        self._worker.gcs_client.internal_kv_put(key, pickle.dumps(actor_class_info), True, KV_NAMESPACE_FUNCTION_TABLE)

    def load_actor_class(self, job_id, actor_creation_function_descriptor):
        """Load the actor class.
        Args:
            job_id: job ID of the actor.
            actor_creation_function_descriptor: Function descriptor of
                the actor constructor.
        Returns:
            The actor class.
        """
        function_id = actor_creation_function_descriptor.function_id
        actor_class = self._loaded_actor_classes.get(function_id, None)
        if actor_class is None:
            if self._worker.load_code_from_local:
                actor_class = self._load_actor_class_from_local(actor_creation_function_descriptor)
                if actor_class is None:
                    actor_class = self._load_actor_class_from_gcs(job_id, actor_creation_function_descriptor)
            else:
                actor_class = self._load_actor_class_from_gcs(job_id, actor_creation_function_descriptor)
            self._loaded_actor_classes[function_id] = actor_class
            module_name = actor_creation_function_descriptor.module_name
            actor_class_name = actor_creation_function_descriptor.class_name
            actor_methods = inspect.getmembers(actor_class, predicate=is_function_or_method)
            for actor_method_name, actor_method in actor_methods:
                if actor_method_name == '__init__':
                    method_descriptor = actor_creation_function_descriptor
                else:
                    method_descriptor = PythonFunctionDescriptor(module_name, actor_method_name, actor_class_name)
                method_id = method_descriptor.function_id
                executor = self._make_actor_method_executor(actor_method_name, actor_method, actor_imported=True)
                self._function_execution_info[method_id] = FunctionExecutionInfo(function=executor, function_name=actor_method_name, max_calls=0)
                self._num_task_executions[method_id] = 0
            self._num_task_executions[function_id] = 0
        return actor_class

    def _load_actor_class_from_local(self, actor_creation_function_descriptor):
        """Load actor class from local code."""
        module_name, class_name = (actor_creation_function_descriptor.module_name, actor_creation_function_descriptor.class_name)
        object = self.load_function_or_class_from_local(module_name, class_name)
        if object is not None:
            if isinstance(object, ray.actor.ActorClass):
                return object.__ray_metadata__.modified_class
            else:
                return object
        else:
            return None

    def _create_fake_actor_class(self, actor_class_name, actor_method_names, traceback_str):

        class TemporaryActor:
            pass

        def temporary_actor_method(*args, **kwargs):
            raise RuntimeError(f'The actor with name {actor_class_name} failed to import on the worker. This may be because needed library dependencies are not installed in the worker environment:\n\n{traceback_str}')
        for method in actor_method_names:
            setattr(TemporaryActor, method, temporary_actor_method)
        return TemporaryActor

    def _load_actor_class_from_gcs(self, job_id, actor_creation_function_descriptor):
        """Load actor class from GCS."""
        key = make_function_table_key(b'ActorClass', job_id, actor_creation_function_descriptor.function_id.binary())
        vals = self._worker.gcs_client.internal_kv_get(key, KV_NAMESPACE_FUNCTION_TABLE)
        fields = ['job_id', 'class_name', 'module', 'class', 'actor_method_names']
        if vals is None:
            vals = {}
        else:
            vals = pickle.loads(vals)
        job_id_str, class_name, module, pickled_class, actor_method_names = (vals.get(field) for field in fields)
        class_name = ensure_str(class_name)
        module_name = ensure_str(module)
        job_id = ray.JobID(job_id_str)
        actor_method_names = json.loads(ensure_str(actor_method_names))
        actor_class = None
        try:
            with self.lock:
                actor_class = pickle.loads(pickled_class)
        except Exception:
            logger.debug('Failed to load actor class %s.', class_name)
            traceback_str = format_error_message(traceback.format_exc())
            actor_class = self._create_fake_actor_class(class_name, actor_method_names, traceback_str)
        actor_class.__module__ = module_name
        return actor_class

    def _make_actor_method_executor(self, method_name: str, method, actor_imported: bool):
        """Make an executor that wraps a user-defined actor method.
        The wrapped method updates the worker's internal state and performs any
        necessary checkpointing operations.
        Args:
            method_name: The name of the actor method.
            method: The actor method to wrap. This should be a
                method defined on the actor class and should therefore take an
                instance of the actor as the first argument.
            actor_imported: Whether the actor has been imported.
                Checkpointing operations will not be run if this is set to
                False.
        Returns:
            A function that executes the given actor method on the worker's
                stored instance of the actor. The function also updates the
                worker's internal state to record the executed method.
        """

        def actor_method_executor(__ray_actor, *args, **kwargs):
            is_bound = is_class_method(method) or is_static_method(type(__ray_actor), method_name)
            if is_bound:
                return method(*args, **kwargs)
            else:
                return method(__ray_actor, *args, **kwargs)
        actor_method_executor.name = method_name
        actor_method_executor.method = method
        return actor_method_executor