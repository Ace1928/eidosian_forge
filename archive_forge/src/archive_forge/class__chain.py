import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from functools import partial as _partial
from functools import reduce
from operator import itemgetter
from types import GeneratorType
from kombu.utils.functional import fxrange, reprcall
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import barrier
from celery._state import current_app
from celery.exceptions import CPendingDeprecationWarning
from celery.result import GroupResult, allow_join_result
from celery.utils import abstract
from celery.utils.collections import ChainMap
from celery.utils.functional import _regen
from celery.utils.functional import chunks as _chunks
from celery.utils.functional import is_list, maybe_list, regen, seq_concat_item, seq_concat_seq
from celery.utils.objects import getitem_property
from celery.utils.text import remove_repeating_from_task, truncate
@Signature.register_type(name='chain')
class _chain(Signature):
    tasks = getitem_property('kwargs.tasks', 'Tasks in chain.')

    @classmethod
    def from_dict(cls, d, app=None):
        tasks = d['kwargs']['tasks']
        if tasks:
            if isinstance(tasks, tuple):
                tasks = d['kwargs']['tasks'] = list(tasks)
            tasks = [maybe_signature(task, app=app) for task in tasks]
        return cls(tasks, app=app, **d['options'])

    def __init__(self, *tasks, **options):
        tasks = regen(tasks[0]) if len(tasks) == 1 and is_list(tasks[0]) else tasks
        super().__init__('celery.chain', (), {'tasks': tasks}, **options)
        self._use_link = options.pop('use_link', None)
        self.subtask_type = 'chain'
        self._frozen = None

    def __call__(self, *args, **kwargs):
        if self.tasks:
            return self.apply_async(args, kwargs)

    def __or__(self, other):
        if isinstance(other, group):
            other = maybe_unroll_group(other)
            if not isinstance(other, group):
                return self.__or__(other)
            tasks = self.unchain_tasks()
            if not tasks:
                return other
            if isinstance(tasks[-1], chord):
                tasks[-1].body = tasks[-1].body | other
                return type(self)(tasks, app=self.app)
            return type(self)(seq_concat_item(tasks, other), app=self._app)
        elif isinstance(other, _chain):
            return type(self)(seq_concat_seq(self.unchain_tasks(), other.unchain_tasks()), app=self._app)
        elif isinstance(other, Signature):
            if self.tasks and isinstance(self.tasks[-1], group):
                sig = self.clone()
                sig.tasks[-1] = chord(sig.tasks[-1], other, app=self._app)
                if len(sig.tasks) > 1 and isinstance(sig.tasks[-2], chord):
                    sig.tasks[-2].body = sig.tasks[-2].body | sig.tasks[-1]
                    sig.tasks = sig.tasks[:-1]
                return sig
            elif self.tasks and isinstance(self.tasks[-1], chord):
                sig = self.clone()
                sig.tasks[-1].body = sig.tasks[-1].body | other
                return sig
            else:
                return type(self)(seq_concat_item(self.unchain_tasks(), other), app=self._app)
        else:
            return NotImplemented

    def clone(self, *args, **kwargs):
        to_signature = maybe_signature
        signature = super().clone(*args, **kwargs)
        signature.kwargs['tasks'] = [to_signature(sig, app=self._app, clone=True) for sig in signature.kwargs['tasks']]
        return signature

    def unchain_tasks(self):
        """Return a list of tasks in the chain.

        The tasks list would be cloned from the chain's tasks.
        All of the chain callbacks would be added to the last task in the (cloned) chain.
        All of the tasks would be linked to the same error callback
        as the chain itself, to ensure that the correct error callback is called
        if any of the (cloned) tasks of the chain fail.
        """
        tasks = [t.clone() for t in self.tasks]
        for sig in maybe_list(self.options.get('link')) or []:
            tasks[-1].link(sig)
        for sig in maybe_list(self.options.get('link_error')) or []:
            for task in tasks:
                task.link_error(sig)
        return tasks

    def apply_async(self, args=None, kwargs=None, **options):
        args = args if args else ()
        kwargs = kwargs if kwargs else []
        app = self.app
        if app.conf.task_always_eager:
            with allow_join_result():
                return self.apply(args, kwargs, **options)
        return self.run(args, kwargs, app=app, **dict(self.options, **options) if options else self.options)

    def run(self, args=None, kwargs=None, group_id=None, chord=None, task_id=None, link=None, link_error=None, publisher=None, producer=None, root_id=None, parent_id=None, app=None, group_index=None, **options):
        """Executes the chain.

        Responsible for executing the chain in the correct order.
        In a case of a chain of a single task, the task is executed directly
        and the result is returned for that task specifically.
        """
        args = args if args else ()
        kwargs = kwargs if kwargs else []
        app = app or self.app
        use_link = self._use_link
        if use_link is None and app.conf.task_protocol == 1:
            use_link = True
        args = tuple(args) + tuple(self.args) if args and (not self.immutable) else self.args
        tasks, results_from_prepare = self.prepare_steps(args, kwargs, self.tasks, root_id, parent_id, link_error, app, task_id, group_id, chord, group_index=group_index)
        if results_from_prepare:
            if link:
                tasks[0].extend_list_option('link', link)
            first_task = tasks.pop()
            options = _prepare_chain_from_options(options, tasks, use_link)
            result_from_apply = first_task.apply_async(**options)
            if not tasks:
                return result_from_apply
            else:
                return results_from_prepare[0]

    def freeze(self, _id=None, group_id=None, chord=None, root_id=None, parent_id=None, group_index=None):
        _, results = self._frozen = self.prepare_steps(self.args, self.kwargs, self.tasks, root_id, parent_id, None, self.app, _id, group_id, chord, clone=False, group_index=group_index)
        return results[0]

    def stamp(self, visitor=None, append_stamps=False, **headers):
        visitor_headers = None
        if visitor is not None:
            visitor_headers = visitor.on_chain_start(self, **headers) or {}
        headers = self._stamp_headers(visitor_headers, append_stamps, **headers)
        self.stamp_links(visitor, **headers)
        for task in self.tasks:
            task.stamp(visitor, append_stamps, **headers)
        if visitor is not None:
            visitor.on_chain_end(self, **headers)

    def prepare_steps(self, args, kwargs, tasks, root_id=None, parent_id=None, link_error=None, app=None, last_task_id=None, group_id=None, chord_body=None, clone=True, from_dict=Signature.from_dict, group_index=None):
        """Prepare the chain for execution.

        To execute a chain, we first need to unpack it correctly.
        During the unpacking, we might encounter other chains, groups, or chords
        which we need to unpack as well.

        For example:
        chain(signature1, chain(signature2, signature3)) --> Upgrades to chain(signature1, signature2, signature3)
        chain(group(signature1, signature2), signature3) --> Upgrades to chord([signature1, signature2], signature3)

        The responsibility of this method is to ensure that the chain is
        correctly unpacked, and then the correct callbacks are set up along the way.

        Arguments:
            args (Tuple): Partial args to be prepended to the existing args.
            kwargs (Dict): Partial kwargs to be merged with existing kwargs.
            tasks (List[Signature]): The tasks of the chain.
            root_id (str): The id of the root task.
            parent_id (str): The id of the parent task.
            link_error (Union[List[Signature], Signature]): The error callback.
                will be set for all tasks in the chain.
            app (Celery): The Celery app instance.
            last_task_id (str): The id of the last task in the chain.
            group_id (str): The id of the group that the chain is a part of.
            chord_body (Signature): The body of the chord, used to synchronize with the chain's
                last task and the chord's body when used together.
            clone (bool): Whether to clone the chain's tasks before modifying them.
            from_dict (Callable): A function that takes a dict and returns a Signature.

        Returns:
            Tuple[List[Signature], List[AsyncResult]]: The frozen tasks of the chain, and the async results
        """
        app = app or self.app
        use_link = self._use_link
        if use_link is None and app.conf.task_protocol == 1:
            use_link = True
        steps = deque(tasks)
        steps_pop = steps.pop
        steps_extend = steps.extend
        prev_task = None
        prev_res = None
        tasks, results = ([], [])
        i = 0
        while steps:
            task = steps_pop()
            is_first_task, is_last_task = (not steps, not i)
            if not isinstance(task, abstract.CallableSignature):
                task = from_dict(task, app=app)
            if isinstance(task, group):
                task = maybe_unroll_group(task)
            if clone:
                if is_first_task:
                    task = task.clone(args, kwargs)
                else:
                    task = task.clone()
            elif is_first_task:
                task.args = tuple(args) + tuple(task.args)
            if isinstance(task, _chain):
                steps_extend(task.tasks)
                continue
            if isinstance(task, group) and prev_task:
                tasks.pop()
                results.pop()
                try:
                    task = chord(task, body=prev_task, task_id=prev_res.task_id, root_id=root_id, app=app)
                except AttributeError:
                    task = chord(task, body=prev_task, root_id=root_id, app=app)
                if tasks:
                    prev_task = tasks[-1]
                    prev_res = results[-1]
                else:
                    prev_task = None
                    prev_res = None
            if is_last_task:
                res = task.freeze(last_task_id, root_id=root_id, group_id=group_id, chord=chord_body, group_index=group_index)
            else:
                res = task.freeze(root_id=root_id)
            i += 1
            if prev_task:
                if use_link:
                    task.link(prev_task)
                if prev_res and (not prev_res.parent):
                    prev_res.parent = res
            if link_error:
                for errback in maybe_list(link_error):
                    task.link_error(errback)
            tasks.append(task)
            results.append(res)
            prev_task, prev_res = (task, res)
            if isinstance(task, chord):
                app.backend.ensure_chords_allowed()
                node = res
                while node.parent:
                    node = node.parent
                prev_res = node
        self.id = last_task_id
        return (tasks, results)

    def apply(self, args=None, kwargs=None, **options):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        last, (fargs, fkwargs) = (None, (args, kwargs))
        for task in self.tasks:
            res = task.clone(fargs, fkwargs).apply(last and (last.get(),), **dict(self.options, **options))
            res.parent, last, (fargs, fkwargs) = (last, res, (None, None))
        return last

    @property
    def app(self):
        app = self._app
        if app is None:
            try:
                app = self.tasks[0]._app
            except LookupError:
                pass
        return app or current_app

    def __repr__(self):
        if not self.tasks:
            return f'<{type(self).__name__}@{id(self):#x}: empty>'
        return remove_repeating_from_task(self.tasks[0]['task'], ' | '.join((repr(t) for t in self.tasks)))