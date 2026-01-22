import sys  # @NoMove
import os
from _pydevd_bundle import pydevd_constants
import atexit
import dis
import io
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import itertools
import traceback
import weakref
import getpass as getpass_mod
import functools
import pydevd_file_utils
from _pydev_bundle import pydev_imports, pydev_log
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydev_bundle._pydev_saved_modules import threading, time, thread
from _pydevd_bundle import pydevd_extension_utils, pydevd_frame_utils
from _pydevd_bundle.pydevd_filtering import FilesFiltering, glob_matches_path
from _pydevd_bundle import pydevd_io, pydevd_vm_type, pydevd_defaults
from _pydevd_bundle import pydevd_utils
from _pydevd_bundle import pydevd_runpy
from _pydev_bundle.pydev_console_utils import DebugConsoleStdIn
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import ExceptionBreakpoint, get_exception_breakpoint
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, CMD_STEP_INTO, CMD_SET_BREAK,
from _pydevd_bundle.pydevd_constants import (get_thread_id, get_current_thread_id,
from _pydevd_bundle.pydevd_defaults import PydevdCustomization  # Note: import alias used on pydev_monkey.
from _pydevd_bundle.pydevd_custom_frames import CustomFramesContainer, custom_frames_container_init
from _pydevd_bundle.pydevd_dont_trace_files import DONT_TRACE, PYDEV_FILE, LIB_FILE, DONT_TRACE_DIRS
from _pydevd_bundle.pydevd_extension_api import DebuggerEventHandler
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, remove_exception_from_frame
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_trace_dispatch import (
from _pydevd_bundle.pydevd_utils import save_main_module, is_current_thread_main_thread, \
from _pydevd_frame_eval.pydevd_frame_eval_main import (
import pydev_ipython  # @UnusedImport
from _pydevd_bundle.pydevd_source_mapping import SourceMapping
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import ThreadingLogger, AsyncioLogger, send_concurrency_message, cur_time
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import wrap_threads
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from pydevd_file_utils import get_fullname, get_package_dir
from os.path import abspath as os_path_abspath
import pydevd_tracing
from _pydevd_bundle.pydevd_comm import (InternalThreadCommand, InternalThreadCommandForAnyThread,
from _pydevd_bundle.pydevd_comm import(InternalConsoleExec,
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread, mark_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_process_net_command_json import PyDevJsonCommandProcessor
from _pydevd_bundle.pydevd_process_net_command import process_net_command
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info, collect_return_info, collect_try_except_info_from_source
from _pydevd_bundle.pydevd_suspended_frames import SuspendedFramesManager
from socket import SHUT_RDWR
from _pydevd_bundle.pydevd_api import PyDevdAPI
from _pydevd_bundle.pydevd_timeout import TimeoutTracker
from _pydevd_bundle.pydevd_thread_lifecycle import suspend_all_threads, mark_thread_suspended
from _pydevd_bundle.pydevd_plugin_utils import PluginManager
class PyDB(object):
    """ Main debugging class
    Lots of stuff going on here:

    PyDB starts two threads on startup that connect to remote debugger (RDB)
    The threads continuously read & write commands to RDB.
    PyDB communicates with these threads through command queues.
       Every RDB command is processed by calling process_net_command.
       Every PyDB net command is sent to the net by posting NetCommand to WriterThread queue

       Some commands need to be executed on the right thread (suspend/resume & friends)
       These are placed on the internal command queue.
    """
    dont_terminate_child_pids = set()

    def __init__(self, set_as_global=True):
        if set_as_global:
            pydevd_tracing.replace_sys_set_trace_func()
        self.authentication = _Authentication()
        self.reader = None
        self.writer = None
        self._fsnotify_thread = None
        self.created_pydb_daemon_threads = {}
        self._waiting_for_connection_thread = None
        self._on_configuration_done_event = threading.Event()
        self.check_alive_thread = None
        self.py_db_command_thread = None
        self.quitting = None
        self.cmd_factory = NetCommandFactory()
        self._cmd_queue = defaultdict(_queue.Queue)
        self.suspended_frames_manager = SuspendedFramesManager()
        self._files_filtering = FilesFiltering()
        self.timeout_tracker = TimeoutTracker(self)
        self.source_mapping = SourceMapping(on_source_mapping_changed=self._clear_filters_caches)
        self.terminate_child_processes = True
        self.terminate_keyboard_interrupt = False
        self.keyboard_interrupt_requested = False
        self.api_received_breakpoints = {}
        self.breakpoints = {}
        self.function_breakpoint_name_to_breakpoint = {}
        PyDevdAPI().set_protocol(self, 0, PydevdCustomization.DEFAULT_PROTOCOL)
        self.variable_presentation = PyDevdAPI.VariablePresentation()
        self.mtime = 0
        self.file_to_id_to_line_breakpoint = {}
        self.file_to_id_to_plugin_breakpoint = {}
        self.break_on_uncaught_exceptions = {}
        self.break_on_caught_exceptions = {}
        self.break_on_user_uncaught_exceptions = {}
        self.ready_to_run = False
        self._main_lock = thread.allocate_lock()
        self._lock_running_thread_ids = thread.allocate_lock()
        self._lock_create_fs_notify = thread.allocate_lock()
        self._py_db_command_thread_event = threading.Event()
        if set_as_global:
            CustomFramesContainer._py_db_command_thread_event = self._py_db_command_thread_event
        self.pydb_disposed = False
        self._wait_for_threads_to_finish_called = False
        self._wait_for_threads_to_finish_called_lock = thread.allocate_lock()
        self._wait_for_threads_to_finish_called_event = threading.Event()
        self.terminate_requested = False
        self._disposed_lock = thread.allocate_lock()
        self.signature_factory = None
        self.SetTrace = pydevd_tracing.SetTrace
        self.skip_on_exceptions_thrown_in_same_context = False
        self.ignore_exceptions_thrown_in_lines_with_ignore_exception = True
        self.skip_suspend_on_breakpoint_exception = ()
        self.skip_print_breakpoint_exception = ()
        self.disable_property_trace = False
        self.disable_property_getter_trace = False
        self.disable_property_setter_trace = False
        self.disable_property_deleter_trace = False
        self._running_thread_ids = {}
        self._enable_thread_notifications = False
        self._set_breakpoints_with_id = False
        self.filename_to_lines_where_exceptions_are_ignored = {}
        self.plugin = None
        self.has_plugin_line_breaks = False
        self.has_plugin_exception_breaks = False
        self.thread_analyser = None
        self.asyncio_analyser = None
        self._gui_event_loop = 'matplotlib'
        self._installed_gui_support = False
        self.gui_in_use = False
        self.activate_gui_function = None
        self.mpl_hooks_in_debug_console = False
        self.mpl_modules_for_patching = {}
        self._filename_to_not_in_scope = {}
        self.first_breakpoint_reached = False
        self._exclude_filters_enabled = self._files_filtering.use_exclude_filters()
        self._is_libraries_filter_enabled = self._files_filtering.use_libraries_filter()
        self.is_files_filter_enabled = self._exclude_filters_enabled or self._is_libraries_filter_enabled
        self.show_return_values = False
        self.remove_return_values_flag = False
        self.redirect_output = False
        self.is_output_redirected = False
        self.use_frame_eval = True
        self._threads_suspended_single_notification = ThreadsSuspendedSingleNotification(self)
        self.stepping_resumes_all_threads = False
        self._local_thread_trace_func = threading.local()
        self._server_socket_ready_event = threading.Event()
        self._server_socket_name = None
        if IS_IRONPYTHON:

            def new_trace_dispatch(frame, event, arg):
                return _trace_dispatch(self, frame, event, arg)
            self.trace_dispatch = new_trace_dispatch
        else:
            self.trace_dispatch = partial(_trace_dispatch, self)
        self.fix_top_level_trace_and_get_trace_func = fix_top_level_trace_and_get_trace_func
        self.frame_eval_func = frame_eval_func
        self.dummy_trace_dispatch = dummy_trace_dispatch
        try:
            self.threading_get_ident = threading.get_ident
            self.threading_active = threading._active
        except:
            try:
                self.threading_get_ident = threading._get_ident
                self.threading_active = threading._active
            except:
                self.threading_get_ident = None
                self.threading_active = None
        self.threading_current_thread = threading.currentThread
        self.set_additional_thread_info = set_additional_thread_info
        self.stop_on_unhandled_exception = stop_on_unhandled_exception
        self.collect_return_info = collect_return_info
        self.get_exception_breakpoint = get_exception_breakpoint
        self._dont_trace_get_file_type = DONT_TRACE.get
        self._dont_trace_dirs_get_file_type = DONT_TRACE_DIRS.get
        self.PYDEV_FILE = PYDEV_FILE
        self.LIB_FILE = LIB_FILE
        self._in_project_scope_cache = {}
        self._exclude_by_filter_cache = {}
        self._apply_filter_cache = {}
        self._ignore_system_exit_codes = set()
        self._dap_messages_listeners = []
        if set_as_global:
            set_global_debugger(self)
        pydevd_defaults.on_pydb_init(self)
        atexit.register(stoptrace)

    def collect_try_except_info(self, code_obj):
        filename = code_obj.co_filename
        try:
            if os.path.exists(filename):
                pydev_log.debug('Collecting try..except info from source for %s', filename)
                try_except_infos = collect_try_except_info_from_source(filename)
                if try_except_infos:
                    max_line = -1
                    min_line = sys.maxsize
                    for _, line in dis.findlinestarts(code_obj):
                        if line > max_line:
                            max_line = line
                        if line < min_line:
                            min_line = line
                    try_except_infos = [x for x in try_except_infos if min_line <= x.try_line <= max_line]
                return try_except_infos
        except:
            pydev_log.exception('Error collecting try..except info from source (%s)', filename)
        pydev_log.debug('Collecting try..except info from bytecode for %s', filename)
        return collect_try_except_info(code_obj)

    def setup_auto_reload_watcher(self, enable_auto_reload, watch_dirs, poll_target_time, exclude_patterns, include_patterns):
        try:
            with self._lock_create_fs_notify:
                if self._fsnotify_thread is not None:
                    self._fsnotify_thread.do_kill_pydev_thread()
                    self._fsnotify_thread = None
                if not enable_auto_reload:
                    return
                exclude_patterns = tuple(exclude_patterns)
                include_patterns = tuple(include_patterns)

                def accept_directory(absolute_filename, cache={}):
                    try:
                        return cache[absolute_filename]
                    except:
                        if absolute_filename and absolute_filename[-1] not in ('/', '\\'):
                            absolute_filename += os.path.sep
                        for include_pattern in include_patterns:
                            if glob_matches_path(absolute_filename, include_pattern):
                                cache[absolute_filename] = True
                                return True
                        for exclude_pattern in exclude_patterns:
                            if glob_matches_path(absolute_filename, exclude_pattern):
                                cache[absolute_filename] = False
                                return False
                        cache[absolute_filename] = True
                        return True

                def accept_file(absolute_filename, cache={}):
                    try:
                        return cache[absolute_filename]
                    except:
                        for include_pattern in include_patterns:
                            if glob_matches_path(absolute_filename, include_pattern):
                                cache[absolute_filename] = True
                                return True
                        for exclude_pattern in exclude_patterns:
                            if glob_matches_path(absolute_filename, exclude_pattern):
                                cache[absolute_filename] = False
                                return False
                        cache[absolute_filename] = False
                        return False
                self._fsnotify_thread = FSNotifyThread(self, PyDevdAPI(), watch_dirs)
                watcher = self._fsnotify_thread.watcher
                watcher.accept_directory = accept_directory
                watcher.accept_file = accept_file
                watcher.target_time_for_single_scan = poll_target_time
                watcher.target_time_for_notification = poll_target_time
                self._fsnotify_thread.start()
        except:
            pydev_log.exception('Error setting up auto-reload.')

    def get_arg_ppid(self):
        try:
            setup = SetupHolder.setup
            if setup:
                return int(setup.get('ppid', 0))
        except:
            pydev_log.exception('Error getting ppid.')
        return 0

    def wait_for_ready_to_run(self):
        while not self.ready_to_run:
            self.process_internal_commands()
            self._py_db_command_thread_event.clear()
            self._py_db_command_thread_event.wait(0.1)

    def on_initialize(self):
        """
        Note: only called when using the DAP (Debug Adapter Protocol).
        """
        self._on_configuration_done_event.clear()

    def on_configuration_done(self):
        """
        Note: only called when using the DAP (Debug Adapter Protocol).
        """
        self._on_configuration_done_event.set()
        self._py_db_command_thread_event.set()

    def is_attached(self):
        return self._on_configuration_done_event.is_set()

    def on_disconnect(self):
        """
        Note: only called when using the DAP (Debug Adapter Protocol).
        """
        self.authentication.logout()
        self._on_configuration_done_event.clear()

    def set_ignore_system_exit_codes(self, ignore_system_exit_codes):
        assert isinstance(ignore_system_exit_codes, (list, tuple, set))
        self._ignore_system_exit_codes = set(ignore_system_exit_codes)

    def ignore_system_exit_code(self, system_exit_exc):
        if hasattr(system_exit_exc, 'code'):
            return system_exit_exc.code in self._ignore_system_exit_codes
        else:
            return system_exit_exc in self._ignore_system_exit_codes

    def block_until_configuration_done(self, cancel=None):
        if cancel is None:
            cancel = NULL
        while not cancel.is_set():
            if self._on_configuration_done_event.is_set():
                cancel.set()
                return
            self.process_internal_commands()
            self._py_db_command_thread_event.clear()
            self._py_db_command_thread_event.wait(1 / 15.0)

    def add_fake_frame(self, thread_id, frame_id, frame):
        self.suspended_frames_manager.add_fake_frame(thread_id, frame_id, frame)

    def handle_breakpoint_condition(self, info, pybreakpoint, new_frame):
        condition = pybreakpoint.condition
        try:
            if pybreakpoint.handle_hit_condition(new_frame):
                return True
            if not condition:
                return False
            return eval(condition, new_frame.f_globals, new_frame.f_locals)
        except Exception as e:
            if not isinstance(e, self.skip_print_breakpoint_exception):
                stack_trace = io.StringIO()
                etype, value, tb = sys.exc_info()
                traceback.print_exception(etype, value, tb.tb_next, file=stack_trace)
                msg = 'Error while evaluating expression in conditional breakpoint: %s\n%s' % (condition, stack_trace.getvalue())
                api = PyDevdAPI()
                api.send_error_message(self, msg)
            if not isinstance(e, self.skip_suspend_on_breakpoint_exception):
                try:
                    etype, value, tb = sys.exc_info()
                    error = ''.join(traceback.format_exception_only(etype, value))
                    stack = traceback.extract_stack(f=tb.tb_frame.f_back)
                    info.conditional_breakpoint_exception = ('Condition:\n' + condition + '\n\nError:\n' + error, stack)
                except:
                    pydev_log.exception()
                return True
            return False
        finally:
            etype, value, tb = (None, None, None)

    def handle_breakpoint_expression(self, pybreakpoint, info, new_frame):
        try:
            try:
                val = eval(pybreakpoint.expression, new_frame.f_globals, new_frame.f_locals)
            except:
                val = sys.exc_info()[1]
        finally:
            if val is not None:
                info.pydev_message = str(val)

    def _internal_get_file_type(self, abs_real_path_and_basename):
        basename = abs_real_path_and_basename[-1]
        if basename.startswith(IGNORE_BASENAMES_STARTING_WITH) or abs_real_path_and_basename[0].startswith(IGNORE_BASENAMES_STARTING_WITH):
            return self.PYDEV_FILE
        file_type = self._dont_trace_get_file_type(basename)
        if file_type is not None:
            return file_type
        if basename.startswith('__init__.py'):
            abs_path = abs_real_path_and_basename[0]
            i = max(abs_path.rfind('/'), abs_path.rfind('\\'))
            if i:
                abs_path = abs_path[0:i]
                i = max(abs_path.rfind('/'), abs_path.rfind('\\'))
                if i:
                    dirname = abs_path[i + 1:]
                    return self._dont_trace_dirs_get_file_type(dirname)
        return None

    def dont_trace_external_files(self, abs_path):
        """
        :param abs_path:
            The result from get_abs_path_real_path_and_base_from_file or
            get_abs_path_real_path_and_base_from_frame.

        :return
            True :
                If files should NOT be traced.

            False:
                If files should be traced.
        """
        return False

    def get_file_type(self, frame, abs_real_path_and_basename=None, _cache_file_type=_CACHE_FILE_TYPE):
        """
        :param abs_real_path_and_basename:
            The result from get_abs_path_real_path_and_base_from_file or
            get_abs_path_real_path_and_base_from_frame.

        :return
            _pydevd_bundle.pydevd_dont_trace_files.PYDEV_FILE:
                If it's a file internal to the debugger which shouldn't be
                traced nor shown to the user.

            _pydevd_bundle.pydevd_dont_trace_files.LIB_FILE:
                If it's a file in a library which shouldn't be traced.

            None:
                If it's a regular user file which should be traced.
        """
        if abs_real_path_and_basename is None:
            try:
                abs_real_path_and_basename = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
            except:
                abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)
        cache_key = (frame.f_code.co_firstlineno, abs_real_path_and_basename[0], frame.f_code)
        try:
            return _cache_file_type[cache_key]
        except:
            if abs_real_path_and_basename[0] == '<string>':
                f = frame.f_back
                while f is not None:
                    if self.get_file_type(f) != self.PYDEV_FILE and pydevd_file_utils.basename(f.f_code.co_filename) not in ('runpy.py', '<string>'):
                        _cache_file_type[cache_key] = LIB_FILE
                        return LIB_FILE
                    f = f.f_back
                else:
                    _cache_file_type[cache_key] = None
                    return None
            file_type = self._internal_get_file_type(abs_real_path_and_basename)
            if file_type is None:
                if self.dont_trace_external_files(abs_real_path_and_basename[0]):
                    file_type = PYDEV_FILE
            _cache_file_type[cache_key] = file_type
            return file_type

    def is_cache_file_type_empty(self):
        return not _CACHE_FILE_TYPE

    def get_cache_file_type(self, _cache=_CACHE_FILE_TYPE):
        return _cache

    def get_thread_local_trace_func(self):
        try:
            thread_trace_func = self._local_thread_trace_func.thread_trace_func
        except AttributeError:
            thread_trace_func = self.trace_dispatch
        return thread_trace_func

    def enable_tracing(self, thread_trace_func=None, apply_to_all_threads=False):
        """
        Enables tracing.

        If in regular mode (tracing), will set the tracing function to the tracing
        function for this thread -- by default it's `PyDB.trace_dispatch`, but after
        `PyDB.enable_tracing` is called with a `thread_trace_func`, the given function will
        be the default for the given thread.

        :param bool apply_to_all_threads:
            If True we'll set the tracing function in all threads, not only in the current thread.
            If False only the tracing for the current function should be changed.
            In general apply_to_all_threads should only be true if this is the first time
            this function is called on a multi-threaded program (either programmatically or attach
            to pid).
        """
        if pydevd_gevent_integration is not None:
            pydevd_gevent_integration.enable_gevent_integration()
        if self.frame_eval_func is not None:
            self.frame_eval_func()
            pydevd_tracing.SetTrace(self.dummy_trace_dispatch)
            if IS_CPYTHON and apply_to_all_threads:
                pydevd_tracing.set_trace_to_threads(self.dummy_trace_dispatch)
            return
        if apply_to_all_threads:
            assert thread_trace_func is not None
        elif thread_trace_func is None:
            thread_trace_func = self.get_thread_local_trace_func()
        else:
            self._local_thread_trace_func.thread_trace_func = thread_trace_func
        pydevd_tracing.SetTrace(thread_trace_func)
        if IS_CPYTHON and apply_to_all_threads:
            pydevd_tracing.set_trace_to_threads(thread_trace_func)

    def disable_tracing(self):
        pydevd_tracing.SetTrace(None)

    def on_breakpoints_changed(self, removed=False):
        """
        When breakpoints change, we have to re-evaluate all the assumptions we've made so far.
        """
        if not self.ready_to_run:
            return
        self.mtime += 1
        if not removed:
            self.set_tracing_for_untraced_contexts()

    def set_tracing_for_untraced_contexts(self):
        if IS_CPYTHON:
            tid_to_frame = sys._current_frames()
            ignore_thread_ids = set((t.ident for t in threadingEnumerate() if getattr(t, 'is_pydev_daemon_thread', False) or getattr(t, 'pydev_do_not_trace', False)))
            for thread_id, frame in tid_to_frame.items():
                if thread_id not in ignore_thread_ids:
                    self.set_trace_for_frame_and_parents(frame)
        else:
            try:
                threads = threadingEnumerate()
                for t in threads:
                    if getattr(t, 'is_pydev_daemon_thread', False) or getattr(t, 'pydev_do_not_trace', False):
                        continue
                    additional_info = set_additional_thread_info(t)
                    frame = additional_info.get_topmost_frame(t)
                    try:
                        if frame is not None:
                            self.set_trace_for_frame_and_parents(frame)
                    finally:
                        frame = None
            finally:
                frame = None
                t = None
                threads = None
                additional_info = None

    @property
    def multi_threads_single_notification(self):
        return self._threads_suspended_single_notification.multi_threads_single_notification

    @multi_threads_single_notification.setter
    def multi_threads_single_notification(self, notify):
        self._threads_suspended_single_notification.multi_threads_single_notification = notify

    @property
    def threads_suspended_single_notification(self):
        return self._threads_suspended_single_notification

    def get_plugin_lazy_init(self):
        if self.plugin is None:
            self.plugin = PluginManager(self)
        return self.plugin

    def in_project_scope(self, frame, absolute_filename=None):
        """
        Note: in general this method should not be used (apply_files_filter should be used
        in most cases as it also handles the project scope check).

        :param frame:
            The frame we want to check.

        :param absolute_filename:
            Must be the result from get_abs_path_real_path_and_base_from_frame(frame)[0] (can
            be used to speed this function a bit if it's already available to the caller, but
            in general it's not needed).
        """
        try:
            if absolute_filename is None:
                try:
                    abs_real_path_and_basename = NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
                except:
                    abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)
                absolute_filename = abs_real_path_and_basename[0]
            cache_key = (frame.f_code.co_firstlineno, absolute_filename, frame.f_code)
            return self._in_project_scope_cache[cache_key]
        except KeyError:
            cache = self._in_project_scope_cache
            try:
                abs_real_path_and_basename
            except NameError:
                abs_real_path_and_basename = get_abs_path_real_path_and_base_from_frame(frame)
            file_type = self.get_file_type(frame, abs_real_path_and_basename)
            if file_type == self.PYDEV_FILE:
                cache[cache_key] = False
            elif absolute_filename == '<string>':
                if file_type == self.LIB_FILE:
                    cache[cache_key] = False
                else:
                    cache[cache_key] = True
            elif self.source_mapping.has_mapping_entry(absolute_filename):
                cache[cache_key] = True
            else:
                cache[cache_key] = self._files_filtering.in_project_roots(absolute_filename)
            return cache[cache_key]

    def in_project_roots_filename_uncached(self, absolute_filename):
        return self._files_filtering.in_project_roots(absolute_filename)

    def _clear_filters_caches(self):
        self._in_project_scope_cache.clear()
        self._exclude_by_filter_cache.clear()
        self._apply_filter_cache.clear()
        self._exclude_filters_enabled = self._files_filtering.use_exclude_filters()
        self._is_libraries_filter_enabled = self._files_filtering.use_libraries_filter()
        self.is_files_filter_enabled = self._exclude_filters_enabled or self._is_libraries_filter_enabled

    def clear_dont_trace_start_end_patterns_caches(self):
        self.on_breakpoints_changed()
        _CACHE_FILE_TYPE.clear()
        self._clear_filters_caches()
        self._clear_skip_caches()

    def _exclude_by_filter(self, frame, absolute_filename):
        """
        :return: True if it should be excluded, False if it should be included and None
            if no rule matched the given file.

        :note: it'll be normalized as needed inside of this method.
        """
        cache_key = (absolute_filename, frame.f_code.co_name, frame.f_code.co_firstlineno)
        try:
            return self._exclude_by_filter_cache[cache_key]
        except KeyError:
            cache = self._exclude_by_filter_cache
            if self.get_file_type(frame) == self.PYDEV_FILE:
                cache[cache_key] = True
            else:
                module_name = None
                if self._files_filtering.require_module:
                    module_name = frame.f_globals.get('__name__', '')
                cache[cache_key] = self._files_filtering.exclude_by_filter(absolute_filename, module_name)
            return cache[cache_key]

    def apply_files_filter(self, frame, original_filename, force_check_project_scope):
        """
        Should only be called if `self.is_files_filter_enabled == True` or `force_check_project_scope == True`.

        Note that it covers both the filter by specific paths includes/excludes as well
        as the check which filters out libraries if not in the project scope.

        :param original_filename:
            Note can either be the original filename or the absolute version of that filename.

        :param force_check_project_scope:
            Check that the file is in the project scope even if the global setting
            is off.

        :return bool:
            True if it should be excluded when stepping and False if it should be
            included.
        """
        cache_key = (frame.f_code.co_firstlineno, original_filename, force_check_project_scope, frame.f_code)
        try:
            return self._apply_filter_cache[cache_key]
        except KeyError:
            if self.plugin is not None and (self.has_plugin_line_breaks or self.has_plugin_exception_breaks):
                if not self.plugin.can_skip(self, frame):
                    pydev_log.debug_once('File traced (included by plugins): %s', original_filename)
                    self._apply_filter_cache[cache_key] = False
                    return False
            if self._exclude_filters_enabled:
                absolute_filename = pydevd_file_utils.absolute_path(original_filename)
                exclude_by_filter = self._exclude_by_filter(frame, absolute_filename)
                if exclude_by_filter is not None:
                    if exclude_by_filter:
                        pydev_log.debug_once('File not traced (excluded by filters): %s', original_filename)
                        self._apply_filter_cache[cache_key] = True
                        return True
                    else:
                        pydev_log.debug_once('File traced (explicitly included by filters): %s', original_filename)
                        self._apply_filter_cache[cache_key] = False
                        return False
            if (self._is_libraries_filter_enabled or force_check_project_scope) and (not self.in_project_scope(frame)):
                self._apply_filter_cache[cache_key] = True
                if force_check_project_scope:
                    pydev_log.debug_once('File not traced (not in project): %s', original_filename)
                else:
                    pydev_log.debug_once('File not traced (not in project - force_check_project_scope): %s', original_filename)
                return True
            if force_check_project_scope:
                pydev_log.debug_once('File traced: %s (force_check_project_scope)', original_filename)
            else:
                pydev_log.debug_once('File traced: %s', original_filename)
            self._apply_filter_cache[cache_key] = False
            return False

    def exclude_exception_by_filter(self, exception_breakpoint, trace):
        if not exception_breakpoint.ignore_libraries and (not self._exclude_filters_enabled):
            return False
        if trace is None:
            return True
        ignore_libraries = exception_breakpoint.ignore_libraries
        exclude_filters_enabled = self._exclude_filters_enabled
        if ignore_libraries and (not self.in_project_scope(trace.tb_frame)) or (exclude_filters_enabled and self._exclude_by_filter(trace.tb_frame, pydevd_file_utils.absolute_path(trace.tb_frame.f_code.co_filename))):
            return True
        return False

    def set_project_roots(self, project_roots):
        self._files_filtering.set_project_roots(project_roots)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def set_exclude_filters(self, exclude_filters):
        self._files_filtering.set_exclude_filters(exclude_filters)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def set_use_libraries_filter(self, use_libraries_filter):
        self._files_filtering.set_use_libraries_filter(use_libraries_filter)
        self._clear_skip_caches()
        self._clear_filters_caches()

    def get_use_libraries_filter(self):
        return self._files_filtering.use_libraries_filter()

    def get_require_module_for_filters(self):
        return self._files_filtering.require_module

    def has_user_threads_alive(self):
        for t in pydevd_utils.get_non_pydevd_threads():
            if isinstance(t, PyDBDaemonThread):
                pydev_log.error_once('Error in debugger: Found PyDBDaemonThread not marked with is_pydev_daemon_thread=True.\n')
            if is_thread_alive(t):
                if not t.daemon or hasattr(t, '__pydevd_main_thread'):
                    return True
        return False

    def initialize_network(self, sock, terminate_on_socket_close=True):
        assert sock is not None
        try:
            sock.settimeout(None)
        except:
            pass
        curr_reader = getattr(self, 'reader', None)
        curr_writer = getattr(self, 'writer', None)
        if curr_reader:
            curr_reader.do_kill_pydev_thread()
        if curr_writer:
            curr_writer.do_kill_pydev_thread()
        self.writer = WriterThread(sock, self, terminate_on_socket_close=terminate_on_socket_close)
        self.reader = ReaderThread(sock, self, PyDevJsonCommandProcessor=PyDevJsonCommandProcessor, process_net_command=process_net_command, terminate_on_socket_close=terminate_on_socket_close)
        self.writer.start()
        self.reader.start()
        time.sleep(0.1)

    def connect(self, host, port):
        if host:
            s = start_client(host, port)
        else:
            s = start_server(port)
        self.initialize_network(s)

    def create_wait_for_connection_thread(self):
        if self._waiting_for_connection_thread is not None:
            raise AssertionError('There is already another thread waiting for a connection.')
        self._server_socket_ready_event.clear()
        self._waiting_for_connection_thread = self._WaitForConnectionThread(self)
        self._waiting_for_connection_thread.start()

    def set_server_socket_ready(self):
        self._server_socket_ready_event.set()

    def wait_for_server_socket_ready(self):
        self._server_socket_ready_event.wait()

    @property
    def dap_messages_listeners(self):
        return self._dap_messages_listeners

    def add_dap_messages_listener(self, listener):
        self._dap_messages_listeners.append(listener)

    class _WaitForConnectionThread(PyDBDaemonThread):

        def __init__(self, py_db):
            PyDBDaemonThread.__init__(self, py_db)
            self._server_socket = None

        def run(self):
            host = SetupHolder.setup['client']
            port = SetupHolder.setup['port']
            self._server_socket = create_server_socket(host=host, port=port)
            self.py_db._server_socket_name = self._server_socket.getsockname()
            self.py_db.set_server_socket_ready()
            while not self._kill_received:
                try:
                    s = self._server_socket
                    if s is None:
                        return
                    s.listen(1)
                    new_socket, _addr = s.accept()
                    if self._kill_received:
                        pydev_log.info('Connection (from wait_for_attach) accepted but ignored as kill was already received.')
                        return
                    pydev_log.info('Connection (from wait_for_attach) accepted.')
                    reader = getattr(self.py_db, 'reader', None)
                    if reader is not None:
                        api = PyDevdAPI()
                        api.request_disconnect(self.py_db, resume_threads=False)
                    self.py_db.initialize_network(new_socket, terminate_on_socket_close=False)
                except:
                    if DebugInfoHolder.DEBUG_TRACE_LEVEL > 0:
                        pydev_log.exception()
                        pydev_log.debug('Exiting _WaitForConnectionThread: %s\n', port)

        def do_kill_pydev_thread(self):
            PyDBDaemonThread.do_kill_pydev_thread(self)
            s = self._server_socket
            if s is not None:
                try:
                    s.close()
                except:
                    pass
                self._server_socket = None

    def get_internal_queue(self, thread_id):
        """ returns internal command queue for a given thread.
        if new queue is created, notify the RDB about it """
        if thread_id.startswith('__frame__'):
            thread_id = thread_id[thread_id.rfind('|') + 1:]
        return self._cmd_queue[thread_id]

    def post_method_as_internal_command(self, thread_id, method, *args, **kwargs):
        if thread_id == '*':
            internal_cmd = InternalThreadCommandForAnyThread(thread_id, method, *args, **kwargs)
        else:
            internal_cmd = InternalThreadCommand(thread_id, method, *args, **kwargs)
        self.post_internal_command(internal_cmd, thread_id)
        if thread_id == '*':
            self._py_db_command_thread_event.set()

    def post_internal_command(self, int_cmd, thread_id):
        """ if thread_id is *, post to the '*' queue"""
        queue = self.get_internal_queue(thread_id)
        queue.put(int_cmd)

    def enable_output_redirection(self, redirect_stdout, redirect_stderr):
        global _global_redirect_stdout_to_server
        global _global_redirect_stderr_to_server
        _global_redirect_stdout_to_server = redirect_stdout
        _global_redirect_stderr_to_server = redirect_stderr
        self.redirect_output = redirect_stdout or redirect_stderr
        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()
        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()

    def check_output_redirect(self):
        global _global_redirect_stdout_to_server
        global _global_redirect_stderr_to_server
        if _global_redirect_stdout_to_server:
            _init_stdout_redirect()
        if _global_redirect_stderr_to_server:
            _init_stderr_redirect()

    def init_matplotlib_in_debug_console(self):
        from _pydev_bundle.pydev_import_hook import import_hook_manager
        if is_current_thread_main_thread():
            for module in list(self.mpl_modules_for_patching):
                import_hook_manager.add_module_name(module, self.mpl_modules_for_patching.pop(module))

    def init_gui_support(self):
        if self._installed_gui_support:
            return
        self._installed_gui_support = True

        class _ReturnGuiLoopControlHelper:
            _return_control_osc = False

        def return_control():
            _ReturnGuiLoopControlHelper._return_control_osc = not _ReturnGuiLoopControlHelper._return_control_osc
            return _ReturnGuiLoopControlHelper._return_control_osc
        from pydev_ipython.inputhook import set_return_control_callback, enable_gui
        set_return_control_callback(return_control)
        if self._gui_event_loop == 'matplotlib':
            from pydev_ipython.matplotlibtools import activate_matplotlib, activate_pylab, activate_pyplot, do_enable_gui
            self.mpl_modules_for_patching = {'matplotlib': lambda: activate_matplotlib(do_enable_gui), 'matplotlib.pyplot': activate_pyplot, 'pylab': activate_pylab}
        else:
            self.activate_gui_function = enable_gui

    def _activate_gui_if_needed(self):
        if self.gui_in_use:
            return
        if len(self.mpl_modules_for_patching) > 0:
            if is_current_thread_main_thread():
                for module in list(self.mpl_modules_for_patching):
                    if module in sys.modules:
                        activate_function = self.mpl_modules_for_patching.pop(module, None)
                        if activate_function is not None:
                            activate_function()
                        self.gui_in_use = True
        if self.activate_gui_function:
            if is_current_thread_main_thread():
                try:
                    self.activate_gui_function(self._gui_event_loop)
                    self.activate_gui_function = None
                    self.gui_in_use = True
                except ValueError:
                    from pydev_ipython.inputhook import set_inputhook
                    try:
                        inputhook_function = import_attr_from_module(self._gui_event_loop)
                        set_inputhook(inputhook_function)
                        self.gui_in_use = True
                    except Exception as e:
                        pydev_log.debug('Cannot activate custom GUI event loop {}: {}'.format(self._gui_event_loop, e))
                    finally:
                        self.activate_gui_function = None

    def _call_input_hook(self):
        try:
            from pydev_ipython.inputhook import get_inputhook
            inputhook = get_inputhook()
            if inputhook:
                inputhook()
        except:
            pass

    def notify_skipped_step_in_because_of_filters(self, frame):
        self.writer.add_command(self.cmd_factory.make_skipped_step_in_because_of_filters(self, frame))

    def notify_thread_created(self, thread_id, thread, use_lock=True):
        if self.writer is None:
            return
        with self._lock_running_thread_ids if use_lock else NULL:
            if not self._enable_thread_notifications:
                return
            if thread_id in self._running_thread_ids:
                return
            additional_info = set_additional_thread_info(thread)
            if additional_info.pydev_notify_kill:
                return
            self._running_thread_ids[thread_id] = thread
        self.writer.add_command(self.cmd_factory.make_thread_created_message(thread))

    def notify_thread_not_alive(self, thread_id, use_lock=True):
        """ if thread is not alive, cancel trace_dispatch processing """
        if self.writer is None:
            return
        with self._lock_running_thread_ids if use_lock else NULL:
            if not self._enable_thread_notifications:
                return
            thread = self._running_thread_ids.pop(thread_id, None)
            if thread is None:
                return
            additional_info = set_additional_thread_info(thread)
            was_notified = additional_info.pydev_notify_kill
            if not was_notified:
                additional_info.pydev_notify_kill = True
        self.writer.add_command(self.cmd_factory.make_thread_killed_message(thread_id))

    def set_enable_thread_notifications(self, enable):
        with self._lock_running_thread_ids:
            if self._enable_thread_notifications != enable:
                self._enable_thread_notifications = enable
                if enable:
                    self._running_thread_ids = {}

    def process_internal_commands(self):
        """
        This function processes internal commands.
        """
        ready_to_run = self.ready_to_run
        dispose = False
        with self._main_lock:
            program_threads_alive = {}
            if ready_to_run:
                self.check_output_redirect()
                all_threads = threadingEnumerate()
                program_threads_dead = []
                with self._lock_running_thread_ids:
                    reset_cache = not self._running_thread_ids
                    for t in all_threads:
                        if getattr(t, 'is_pydev_daemon_thread', False):
                            pass
                        elif isinstance(t, PyDBDaemonThread):
                            pydev_log.error_once('Error in debugger: Found PyDBDaemonThread not marked with is_pydev_daemon_thread=True.')
                        elif is_thread_alive(t):
                            if reset_cache:
                                clear_cached_thread_id(t)
                            thread_id = get_thread_id(t)
                            program_threads_alive[thread_id] = t
                            self.notify_thread_created(thread_id, t, use_lock=False)
                    thread_ids = list(self._running_thread_ids.keys())
                    for thread_id in thread_ids:
                        if thread_id not in program_threads_alive:
                            program_threads_dead.append(thread_id)
                    for thread_id in program_threads_dead:
                        self.notify_thread_not_alive(thread_id, use_lock=False)
            cmds_to_execute = []
            if len(program_threads_alive) == 0 and ready_to_run:
                dispose = True
            else:
                curr_thread_id = get_current_thread_id(threadingCurrentThread())
                if ready_to_run:
                    process_thread_ids = (curr_thread_id, '*')
                else:
                    process_thread_ids = ('*',)
                for thread_id in process_thread_ids:
                    queue = self.get_internal_queue(thread_id)
                    cmds_to_add_back = []
                    try:
                        while True:
                            int_cmd = queue.get(False)
                            if not self.mpl_hooks_in_debug_console and isinstance(int_cmd, InternalConsoleExec) and (not self.gui_in_use):
                                try:
                                    self.init_matplotlib_in_debug_console()
                                    self.gui_in_use = True
                                except:
                                    pydev_log.debug('Matplotlib support in debug console failed', traceback.format_exc())
                                self.mpl_hooks_in_debug_console = True
                            if int_cmd.can_be_executed_by(curr_thread_id):
                                cmds_to_execute.append(int_cmd)
                            else:
                                pydev_log.verbose('NOT processing internal command: %s ', int_cmd)
                                cmds_to_add_back.append(int_cmd)
                    except _queue.Empty:
                        for int_cmd in cmds_to_add_back:
                            queue.put(int_cmd)
        if dispose:
            self.dispose_and_kill_all_pydevd_threads()
        else:
            for int_cmd in cmds_to_execute:
                pydev_log.verbose('processing internal command: %s', int_cmd)
                try:
                    int_cmd.do_it(self)
                except:
                    pydev_log.exception('Error processing internal command.')

    def consolidate_breakpoints(self, canonical_normalized_filename, id_to_breakpoint, file_to_line_to_breakpoints):
        break_dict = {}
        for _breakpoint_id, pybreakpoint in id_to_breakpoint.items():
            break_dict[pybreakpoint.line] = pybreakpoint
        file_to_line_to_breakpoints[canonical_normalized_filename] = break_dict
        self._clear_skip_caches()

    def _clear_skip_caches(self):
        global_cache_skips.clear()
        global_cache_frame_skips.clear()

    def add_break_on_exception(self, exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries=False):
        try:
            eb = ExceptionBreakpoint(exception, condition, expression, notify_on_handled_exceptions, notify_on_unhandled_exceptions, notify_on_user_unhandled_exceptions, notify_on_first_raise_only, ignore_libraries)
        except ImportError:
            pydev_log.critical('Error unable to add break on exception for: %s (exception could not be imported).', exception)
            return None
        if eb.notify_on_unhandled_exceptions:
            cp = self.break_on_uncaught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info('Exceptions to hook on terminate: %s.', cp)
            self.break_on_uncaught_exceptions = cp
        if eb.notify_on_handled_exceptions:
            cp = self.break_on_caught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info('Exceptions to hook always: %s.', cp)
            self.break_on_caught_exceptions = cp
        if eb.notify_on_user_unhandled_exceptions:
            cp = self.break_on_user_uncaught_exceptions.copy()
            cp[exception] = eb
            pydev_log.info('Exceptions to hook on user uncaught code: %s.', cp)
            self.break_on_user_uncaught_exceptions = cp
        return eb

    def set_suspend(self, thread, stop_reason, suspend_other_threads=False, is_pause=False, original_step_cmd=-1):
        """
        :param thread:
            The thread which should be suspended.

        :param stop_reason:
            Reason why the thread was suspended.

        :param suspend_other_threads:
            Whether to force other threads to be suspended (i.e.: when hitting a breakpoint
            with a suspend all threads policy).

        :param is_pause:
            If this is a pause to suspend all threads, any thread can be considered as the 'main'
            thread paused.

        :param original_step_cmd:
            If given we may change the stop reason to this.
        """
        self._threads_suspended_single_notification.increment_suspend_time()
        if is_pause:
            self._threads_suspended_single_notification.on_pause()
        info = mark_thread_suspended(thread, stop_reason, original_step_cmd=original_step_cmd)
        if is_pause:
            frame = info.get_topmost_frame(thread)
            if frame is not None:
                try:
                    self.set_trace_for_frame_and_parents(frame)
                finally:
                    frame = None
        if stop_reason == CMD_SET_BREAK and info.conditional_breakpoint_exception is not None:
            conditional_breakpoint_exception_tuple = info.conditional_breakpoint_exception
            info.conditional_breakpoint_exception = None
            self._send_breakpoint_condition_exception(thread, conditional_breakpoint_exception_tuple)
        if not suspend_other_threads and self.multi_threads_single_notification:
            suspend_other_threads = True
        if suspend_other_threads:
            suspend_all_threads(self, except_thread=thread)

    def _send_breakpoint_condition_exception(self, thread, conditional_breakpoint_exception_tuple):
        """If conditional breakpoint raises an exception during evaluation
        send exception details to java
        """
        thread_id = get_thread_id(thread)
        if conditional_breakpoint_exception_tuple and len(conditional_breakpoint_exception_tuple) == 2:
            exc_type, stacktrace = conditional_breakpoint_exception_tuple
            int_cmd = InternalGetBreakpointException(thread_id, exc_type, stacktrace)
            self.post_internal_command(int_cmd, thread_id)

    def send_caught_exception_stack(self, thread, arg, curr_frame_id):
        """Sends details on the exception which was caught (and where we stopped) to the java side.

        arg is: exception type, description, traceback object
        """
        thread_id = get_thread_id(thread)
        int_cmd = InternalSendCurrExceptionTrace(thread_id, arg, curr_frame_id)
        self.post_internal_command(int_cmd, thread_id)

    def send_caught_exception_stack_proceeded(self, thread):
        """Sends that some thread was resumed and is no longer showing an exception trace.
        """
        thread_id = get_thread_id(thread)
        int_cmd = InternalSendCurrExceptionTraceProceeded(thread_id)
        self.post_internal_command(int_cmd, thread_id)
        self.process_internal_commands()

    def send_process_created_message(self):
        """Sends a message that a new process has been created.
        """
        if self.writer is None or self.cmd_factory is None:
            return
        cmd = self.cmd_factory.make_process_created_message()
        self.writer.add_command(cmd)

    def send_process_about_to_be_replaced(self):
        """Sends a message that a new process has been created.
        """
        if self.writer is None or self.cmd_factory is None:
            return
        cmd = self.cmd_factory.make_process_about_to_be_replaced_message()
        if cmd is NULL_NET_COMMAND:
            return
        sent = [False]

        def after_sent(*args, **kwargs):
            sent[0] = True
        cmd.call_after_send(after_sent)
        self.writer.add_command(cmd)
        timeout = 5
        initial_time = time.time()
        while not sent[0]:
            time.sleep(0.05)
            if time.time() - initial_time > timeout:
                pydev_log.critical('pydevd: Sending message related to process being replaced timed-out after %s seconds', timeout)
                break

    def set_next_statement(self, frame, event, func_name, next_line):
        stop = False
        response_msg = ''
        old_line = frame.f_lineno
        if event == 'line' or event == 'exception':
            curr_func_name = frame.f_code.co_name
            if curr_func_name in ('?', '<module>'):
                curr_func_name = ''
            if func_name == '*' or curr_func_name == func_name:
                line = next_line
                frame.f_trace = self.trace_dispatch
                frame.f_lineno = line
                stop = True
            else:
                response_msg = 'jump is available only within the bottom frame'
        return (stop, old_line, response_msg)

    def cancel_async_evaluation(self, thread_id, frame_id):
        with self._main_lock:
            try:
                all_threads = threadingEnumerate()
                for t in all_threads:
                    if getattr(t, 'is_pydev_daemon_thread', False) and hasattr(t, 'cancel_event') and (t.thread_id == thread_id) and (t.frame_id == frame_id):
                        t.cancel_event.set()
            except:
                pydev_log.exception()

    def find_frame(self, thread_id, frame_id):
        """ returns a frame on the thread that has a given frame_id """
        return self.suspended_frames_manager.find_frame(thread_id, frame_id)

    def do_wait_suspend(self, thread, frame, event, arg, exception_type=None):
        """ busy waits until the thread state changes to RUN
        it expects thread's state as attributes of the thread.
        Upon running, processes any outstanding Stepping commands.

        :param exception_type:
            If pausing due to an exception, its type.
        """
        if USE_CUSTOM_SYS_CURRENT_FRAMES_MAP:
            constructed_tid_to_last_frame[thread.ident] = sys._getframe()
        self.process_internal_commands()
        thread_id = get_current_thread_id(thread)
        message = thread.additional_info.pydev_message
        suspend_type = thread.additional_info.trace_suspend_type
        thread.additional_info.trace_suspend_type = 'trace'
        stop_reason = thread.stop_reason
        frames_list = None
        if arg is not None and event == 'exception':
            exc_type, exc_desc, trace_obj = arg
            if trace_obj is not None:
                frames_list = pydevd_frame_utils.create_frames_list_from_traceback(trace_obj, frame, exc_type, exc_desc, exception_type=exception_type)
        if frames_list is None:
            frames_list = pydevd_frame_utils.create_frames_list_from_frame(frame)
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 2:
            pydev_log.debug('PyDB.do_wait_suspend\nname: %s (line: %s)\n file: %s\n event: %s\n arg: %s\n step: %s (original step: %s)\n thread: %s, thread id: %s, id(thread): %s', frame.f_code.co_name, frame.f_lineno, frame.f_code.co_filename, event, arg, constant_to_str(thread.additional_info.pydev_step_cmd), constant_to_str(thread.additional_info.pydev_original_step_cmd), thread, thread_id, id(thread))
            for f in frames_list:
                pydev_log.debug('  Stack: %s, %s, %s', f.f_code.co_filename, f.f_code.co_name, f.f_lineno)
        with self.suspended_frames_manager.track_frames(self) as frames_tracker:
            frames_tracker.track(thread_id, frames_list)
            cmd = frames_tracker.create_thread_suspend_command(thread_id, stop_reason, message, suspend_type)
            self.writer.add_command(cmd)
            with CustomFramesContainer.custom_frames_lock:
                from_this_thread = []
                for frame_custom_thread_id, custom_frame in CustomFramesContainer.custom_frames.items():
                    if custom_frame.thread_id == thread.ident:
                        frames_tracker.track(thread_id, pydevd_frame_utils.create_frames_list_from_frame(custom_frame.frame), frame_custom_thread_id=frame_custom_thread_id)
                        self.writer.add_command(self.cmd_factory.make_custom_frame_created_message(frame_custom_thread_id, custom_frame.name))
                        self.writer.add_command(frames_tracker.create_thread_suspend_command(frame_custom_thread_id, CMD_THREAD_SUSPEND, '', suspend_type))
                    from_this_thread.append(frame_custom_thread_id)
            with self._threads_suspended_single_notification.notify_thread_suspended(thread_id, thread, stop_reason):
                keep_suspended = self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)
        frames_list = None
        if keep_suspended:
            self._threads_suspended_single_notification.increment_suspend_time()
            self.do_wait_suspend(thread, frame, event, arg, exception_type)
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 2:
            pydev_log.debug('Leaving PyDB.do_wait_suspend: %s (%s) %s', thread, thread_id, id(thread))

    def _do_wait_suspend(self, thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker):
        info = thread.additional_info
        info.step_in_initial_location = None
        keep_suspended = False
        with self._main_lock:
            activate_gui = info.pydev_state == STATE_SUSPEND and (not self.pydb_disposed)
        in_main_thread = is_current_thread_main_thread()
        if activate_gui and in_main_thread:
            self._activate_gui_if_needed()
        while True:
            with self._main_lock:
                if info.pydev_state != STATE_SUSPEND or (self.pydb_disposed and (not self.terminate_requested)):
                    break
            if in_main_thread and self.gui_in_use:
                self._call_input_hook()
            self.process_internal_commands()
            time.sleep(0.01)
        self.cancel_async_evaluation(get_current_thread_id(thread), str(id(frame)))
        if info.pydev_step_cmd in (CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
            info.step_in_initial_location = (frame, frame.f_lineno)
            if frame.f_code.co_flags & 128:
                info.pydev_step_cmd = CMD_STEP_INTO_COROUTINE
                info.pydev_step_stop = frame
                self.set_trace_for_frame_and_parents(frame)
            else:
                info.pydev_step_stop = None
                self.set_trace_for_frame_and_parents(frame)
        elif info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_SMART_STEP_INTO):
            info.pydev_step_stop = frame
            self.set_trace_for_frame_and_parents(frame)
        elif info.pydev_step_cmd == CMD_RUN_TO_LINE or info.pydev_step_cmd == CMD_SET_NEXT_STATEMENT:
            info.pydev_step_stop = None
            self.set_trace_for_frame_and_parents(frame)
            stop = False
            response_msg = ''
            try:
                stop, _old_line, response_msg = self.set_next_statement(frame, event, info.pydev_func_name, info.pydev_next_line)
            except ValueError as e:
                response_msg = '%s' % e
            finally:
                seq = info.pydev_message
                cmd = self.cmd_factory.make_set_next_stmnt_status_message(seq, stop, response_msg)
                self.writer.add_command(cmd)
                info.pydev_message = ''
            if stop:
                frames_tracker.untrack_all()
                cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
                self.writer.add_command(cmd)
                info.pydev_state = STATE_SUSPEND
                thread.stop_reason = CMD_SET_NEXT_STATEMENT
                keep_suspended = True
            else:
                info.pydev_original_step_cmd = -1
                info.pydev_step_cmd = -1
                info.pydev_state = STATE_SUSPEND
                thread.stop_reason = CMD_THREAD_SUSPEND
                return self._do_wait_suspend(thread, frame, event, arg, suspend_type, from_this_thread, frames_tracker)
        elif info.pydev_step_cmd in (CMD_STEP_RETURN, CMD_STEP_RETURN_MY_CODE):
            back_frame = frame.f_back
            force_check_project_scope = info.pydev_step_cmd == CMD_STEP_RETURN_MY_CODE
            if force_check_project_scope or self.is_files_filter_enabled:
                while back_frame is not None:
                    if self.apply_files_filter(back_frame, back_frame.f_code.co_filename, force_check_project_scope):
                        frame = back_frame
                        back_frame = back_frame.f_back
                    else:
                        break
            if back_frame is not None:
                info.pydev_step_stop = frame
                self.set_trace_for_frame_and_parents(frame)
            else:
                info.pydev_step_stop = None
                info.pydev_original_step_cmd = -1
                info.pydev_step_cmd = -1
                info.pydev_state = STATE_RUN
        if PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING:
            info.pydev_use_scoped_step_frame = False
            if info.pydev_step_cmd in (CMD_STEP_OVER, CMD_STEP_OVER_MY_CODE, CMD_STEP_INTO, CMD_STEP_INTO_MY_CODE):
                f = frame.f_back
                if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[1]:
                    f = f.f_back
                    if f and f.f_code.co_name == PYDEVD_IPYTHON_CONTEXT[2]:
                        info.pydev_use_scoped_step_frame = True
                        pydev_log.info('Using (ipython) scoped stepping.')
                del f
        del frame
        cmd = self.cmd_factory.make_thread_run_message(get_current_thread_id(thread), info.pydev_step_cmd)
        self.writer.add_command(cmd)
        with CustomFramesContainer.custom_frames_lock:
            for frame_id in from_this_thread:
                self.writer.add_command(self.cmd_factory.make_thread_killed_message(frame_id))
        return keep_suspended

    def do_stop_on_unhandled_exception(self, thread, frame, frames_byid, arg):
        pydev_log.debug('We are stopping in unhandled exception.')
        try:
            add_exception_to_frame(frame, arg)
            self.send_caught_exception_stack(thread, arg, id(frame))
            try:
                self.set_suspend(thread, CMD_ADD_EXCEPTION_BREAK)
                self.do_wait_suspend(thread, frame, 'exception', arg, EXCEPTION_TYPE_UNHANDLED)
            except:
                self.send_caught_exception_stack_proceeded(thread)
        except:
            pydev_log.exception("We've got an error while stopping in unhandled exception: %s.", arg[0])
        finally:
            remove_exception_from_frame(frame)
            frame = None

    def set_trace_for_frame_and_parents(self, frame, **kwargs):
        disable = kwargs.pop('disable', False)
        assert not kwargs
        while frame is not None:
            file_type = self.get_file_type(frame)
            if file_type is None:
                if disable:
                    pydev_log.debug('Disable tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)
                    if frame.f_trace is not None and frame.f_trace is not NO_FTRACE:
                        frame.f_trace = NO_FTRACE
                elif frame.f_trace is not self.trace_dispatch:
                    pydev_log.debug('Set tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)
                    frame.f_trace = self.trace_dispatch
            else:
                pydev_log.debug('SKIP set tracing of frame: %s - %s', frame.f_code.co_filename, frame.f_code.co_name)
            frame = frame.f_back
        del frame

    def _create_pydb_command_thread(self):
        curr_pydb_command_thread = self.py_db_command_thread
        if curr_pydb_command_thread is not None:
            curr_pydb_command_thread.do_kill_pydev_thread()
        new_pydb_command_thread = self.py_db_command_thread = PyDBCommandThread(self)
        new_pydb_command_thread.start()

    def _create_check_output_thread(self):
        curr_output_checker_thread = self.check_alive_thread
        if curr_output_checker_thread is not None:
            curr_output_checker_thread.do_kill_pydev_thread()
        check_alive_thread = self.check_alive_thread = CheckAliveThread(self)
        check_alive_thread.start()

    def start_auxiliary_daemon_threads(self):
        self._create_pydb_command_thread()
        self._create_check_output_thread()

    def __wait_for_threads_to_finish(self, timeout):
        try:
            with self._wait_for_threads_to_finish_called_lock:
                wait_for_threads_to_finish_called = self._wait_for_threads_to_finish_called
                self._wait_for_threads_to_finish_called = True
            if wait_for_threads_to_finish_called:
                self._wait_for_threads_to_finish_called_event.wait(timeout=timeout)
            else:
                try:

                    def get_pydb_daemon_threads_to_wait():
                        pydb_daemon_threads = set(self.created_pydb_daemon_threads)
                        pydb_daemon_threads.discard(self.check_alive_thread)
                        pydb_daemon_threads.discard(threading.current_thread())
                        return pydb_daemon_threads
                    pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads waiting for pydb daemon threads to finish')
                    started_at = time.time()
                    while time.time() < started_at + timeout:
                        if len(get_pydb_daemon_threads_to_wait()) == 0:
                            break
                        time.sleep(1 / 10.0)
                    else:
                        thread_names = [t.name for t in get_pydb_daemon_threads_to_wait()]
                        if thread_names:
                            pydev_log.debug('The following pydb threads may not have finished correctly: %s', ', '.join(thread_names))
                finally:
                    self._wait_for_threads_to_finish_called_event.set()
        except:
            pydev_log.exception()

    def dispose_and_kill_all_pydevd_threads(self, wait=True, timeout=0.5):
        """
        When this method is called we finish the debug session, terminate threads
        and if this was registered as the global instance, unregister it -- afterwards
        it should be possible to create a new instance and set as global to start
        a new debug session.

        :param bool wait:
            If True we'll wait for the threads to be actually finished before proceeding
            (based on the available timeout).
            Note that this must be thread-safe and if one thread is waiting the other thread should
            also wait.
        """
        try:
            back_frame = sys._getframe().f_back
            pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads (called from: File "%s", line %s, in %s)', back_frame.f_code.co_filename, back_frame.f_lineno, back_frame.f_code.co_name)
            back_frame = None
            with self._disposed_lock:
                disposed = self.pydb_disposed
                self.pydb_disposed = True
            if disposed:
                if wait:
                    pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads (already disposed - wait)')
                    self.__wait_for_threads_to_finish(timeout)
                else:
                    pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads (already disposed - no wait)')
                return
            pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads (first call)')
            started_at = time.time()
            while time.time() < started_at + timeout:
                with self._main_lock:
                    writer = self.writer
                    if writer is None or writer.empty():
                        pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads no commands being processed.')
                        break
            else:
                pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads timed out waiting for writer to be empty.')
            pydb_daemon_threads = set(self.created_pydb_daemon_threads)
            for t in pydb_daemon_threads:
                if hasattr(t, 'do_kill_pydev_thread'):
                    pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads killing thread: %s', t)
                    t.do_kill_pydev_thread()
            if wait:
                self.__wait_for_threads_to_finish(timeout)
            else:
                pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads: no wait')
            py_db = get_global_debugger()
            if py_db is self:
                set_global_debugger(None)
        except:
            pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads: exception')
            try:
                if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 3:
                    pydev_log.exception()
            except:
                pass
        finally:
            pydev_log.debug('PyDB.dispose_and_kill_all_pydevd_threads: finished')

    def prepare_to_run(self):
        """ Shared code to prepare debugging by installing traces and registering threads """
        self.patch_threads()
        self.start_auxiliary_daemon_threads()

    def patch_threads(self):
        try:
            threading.settrace(self.trace_dispatch)
        except:
            pass
        from _pydev_bundle.pydev_monkey import patch_thread_modules
        patch_thread_modules()

    def run(self, file, globals=None, locals=None, is_module=False, set_trace=True):
        module_name = None
        entry_point_fn = ''
        if is_module:
            if '' not in sys.path:
                sys.path.insert(0, '')
            file, _, entry_point_fn = file.partition(':')
            module_name = file
            filename = get_fullname(file)
            if filename is None:
                mod_dir = get_package_dir(module_name)
                if mod_dir is None:
                    sys.stderr.write('No module named %s\n' % file)
                    return
                else:
                    filename = get_fullname('%s.__main__' % module_name)
                    if filename is None:
                        sys.stderr.write('No module named %s\n' % file)
                        return
                    else:
                        file = filename
            else:
                file = filename
                mod_dir = os.path.dirname(filename)
                main_py = os.path.join(mod_dir, '__main__.py')
                main_pyc = os.path.join(mod_dir, '__main__.pyc')
                if filename.endswith('__init__.pyc'):
                    if os.path.exists(main_pyc):
                        filename = main_pyc
                    elif os.path.exists(main_py):
                        filename = main_py
                elif filename.endswith('__init__.py'):
                    if os.path.exists(main_pyc) and (not os.path.exists(main_py)):
                        filename = main_pyc
                    elif os.path.exists(main_py):
                        filename = main_py
            sys.argv[0] = filename
        if os.path.isdir(file):
            new_target = os.path.join(file, '__main__.py')
            if os.path.isfile(new_target):
                file = new_target
        m = None
        if globals is None:
            m = save_main_module(file, 'pydevd')
            globals = m.__dict__
            try:
                globals['__builtins__'] = __builtins__
            except NameError:
                pass
        if locals is None:
            locals = globals
        if sys.path[0] != '' and m is not None and m.__file__.startswith(sys.path[0]):
            del sys.path[0]
        if not is_module:
            sys.path.insert(0, os.path.split(os_path_abspath(file))[0])
        if set_trace:
            self.wait_for_ready_to_run()
            self.prepare_to_run()
        t = threadingCurrentThread()
        thread_id = get_current_thread_id(t)
        if self.thread_analyser is not None:
            wrap_threads()
            self.thread_analyser.set_start_time(cur_time())
            send_concurrency_message('threading_event', 0, t.name, thread_id, 'thread', 'start', file, 1, None, parent=thread_id)
        if self.asyncio_analyser is not None:
            send_concurrency_message('asyncio_event', 0, 'Task', 'Task', 'thread', 'stop', file, 1, frame=None, parent=None)
        try:
            if INTERACTIVE_MODE_AVAILABLE:
                self.init_gui_support()
        except:
            pydev_log.exception('Matplotlib support in debugger failed')
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        self.notify_thread_created(thread_id, t)
        if set_trace:
            self.enable_tracing()
        return self._exec(is_module, entry_point_fn, module_name, file, globals, locals)

    def _exec(self, is_module, entry_point_fn, module_name, file, globals, locals):
        """
        This function should have frames tracked by unhandled exceptions (the `_exec` name is important).
        """
        if not is_module:
            globals = pydevd_runpy.run_path(file, globals, '__main__')
        elif entry_point_fn:
            mod = __import__(module_name, level=0, fromlist=[entry_point_fn], globals=globals, locals=locals)
            func = getattr(mod, entry_point_fn)
            func()
        else:
            globals = pydevd_runpy._run_module_as_main(module_name, alter_argv=False)
        return globals

    def wait_for_commands(self, globals):
        self._activate_gui_if_needed()
        thread = threading.current_thread()
        from _pydevd_bundle import pydevd_frame_utils
        frame = pydevd_frame_utils.Frame(None, -1, pydevd_frame_utils.FCode('Console', os.path.abspath(os.path.dirname(__file__))), globals, globals)
        thread_id = get_current_thread_id(thread)
        self.add_fake_frame(thread_id, id(frame), frame)
        cmd = self.cmd_factory.make_show_console_message(self, thread_id, frame)
        if self.writer is not None:
            self.writer.add_command(cmd)
        while True:
            if self.gui_in_use:
                self._call_input_hook()
            self.process_internal_commands()
            time.sleep(0.01)