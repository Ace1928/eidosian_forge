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
@Signature.register_type(name='chord')
class _chord(Signature):
    """Barrier synchronization primitive.

    A chord consists of a header and a body.

    The header is a group of tasks that must complete before the callback is
    called.  A chord is essentially a callback for a group of tasks.

    The body is applied with the return values of all the header
    tasks as a list.

    Example:

        The chord:

        .. code-block:: pycon

            >>> res = chord([add.s(2, 2), add.s(4, 4)])(sum_task.s())

        is effectively :math:`\\Sigma ((2 + 2) + (4 + 4))`:

        .. code-block:: pycon

            >>> res.get()
            12
    """

    @classmethod
    def from_dict(cls, d, app=None):
        """Create a chord signature from a dictionary that represents a chord.

        Example:
            >>> chord_dict = {
                "task": "celery.chord",
                "args": [],
                "kwargs": {
                    "kwargs": {},
                    "header": [
                        {
                            "task": "add",
                            "args": [
                                1,
                                2
                            ],
                            "kwargs": {},
                            "options": {},
                            "subtask_type": None,
                            "immutable": False
                        },
                        {
                            "task": "add",
                            "args": [
                                3,
                                4
                            ],
                            "kwargs": {},
                            "options": {},
                            "subtask_type": None,
                            "immutable": False
                        }
                    ],
                    "body": {
                        "task": "xsum",
                        "args": [],
                        "kwargs": {},
                        "options": {},
                        "subtask_type": None,
                        "immutable": False
                    }
                },
                "options": {},
                "subtask_type": "chord",
                "immutable": False
            }
            >>> chord_sig = chord.from_dict(chord_dict)

        Iterates over the given tasks in the dictionary and convert them to signatures.
        Chord header needs to be defined in d['kwargs']['header'] as a sequence
        of tasks.
        Chord body needs to be defined in d['kwargs']['body'] as a single task.

        The tasks themselves can be dictionaries or signatures (or both).
        """
        options = d.copy()
        args, options['kwargs'] = cls._unpack_args(**options['kwargs'])
        return cls(*args, app=app, **options)

    @staticmethod
    def _unpack_args(header=None, body=None, **kwargs):
        return ((header, body), kwargs)

    def __init__(self, header, body=None, task='celery.chord', args=None, kwargs=None, app=None, **options):
        args = args if args else ()
        kwargs = kwargs if kwargs else {'kwargs': {}}
        super().__init__(task, args, {**kwargs, 'header': _maybe_group(header, app), 'body': maybe_signature(body, app=app)}, app=app, **options)
        self.subtask_type = 'chord'

    def __call__(self, body=None, **options):
        return self.apply_async((), {'body': body} if body else {}, **options)

    def __or__(self, other):
        if not isinstance(other, (group, _chain)) and isinstance(other, Signature):
            sig = self.clone()
            sig.body = sig.body | other
            return sig
        elif isinstance(other, group) and len(other.tasks) == 1:
            other = maybe_unroll_group(other)
            sig = self.clone()
            sig.body = sig.body | other
            return sig
        else:
            return super().__or__(other)

    def freeze(self, _id=None, group_id=None, chord=None, root_id=None, parent_id=None, group_index=None):
        if not isinstance(self.tasks, group):
            self.tasks = group(self.tasks, app=self.app)
        header_result = self.tasks.freeze(parent_id=parent_id, root_id=root_id, chord=self.body)
        self.id = self.tasks.id
        body_result = None
        if self.body:
            body_result = self.body.freeze(_id, root_id=root_id, chord=chord, group_id=group_id, group_index=group_index)
            node = body_result
            seen = set()
            while node:
                if node.id in seen:
                    raise RuntimeError('Recursive result parents')
                seen.add(node.id)
                if node.parent is None:
                    node.parent = header_result
                    break
                node = node.parent
        return body_result

    def stamp(self, visitor=None, append_stamps=False, **headers):
        tasks = self.tasks
        if isinstance(tasks, group):
            tasks = tasks.tasks
        visitor_headers = None
        if visitor is not None:
            visitor_headers = visitor.on_chord_header_start(self, **headers) or {}
        headers = self._stamp_headers(visitor_headers, append_stamps, **headers)
        self.stamp_links(visitor, append_stamps, **headers)
        if isinstance(tasks, _regen):
            tasks.map(_partial(_stamp_regen_task, visitor=visitor, append_stamps=append_stamps, **headers))
        else:
            stamps = headers.copy()
            for task in tasks:
                task.stamp(visitor, append_stamps, **stamps)
        if visitor is not None:
            visitor.on_chord_header_end(self, **headers)
        if visitor is not None and self.body is not None:
            visitor_headers = visitor.on_chord_body(self, **headers) or {}
            headers = self._stamp_headers(visitor_headers, append_stamps, **headers)
            self.body.stamp(visitor, append_stamps, **headers)

    def apply_async(self, args=None, kwargs=None, task_id=None, producer=None, publisher=None, connection=None, router=None, result_cls=None, **options):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        args = tuple(args) + tuple(self.args) if args and (not self.immutable) else self.args
        body = kwargs.pop('body', None) or self.kwargs['body']
        kwargs = dict(self.kwargs['kwargs'], **kwargs)
        body = body.clone(**options)
        app = self._get_app(body)
        tasks = self.tasks.clone() if isinstance(self.tasks, group) else group(self.tasks, app=app, task_id=self.options.get('task_id', uuid()))
        if app.conf.task_always_eager:
            with allow_join_result():
                return self.apply(args, kwargs, body=body, task_id=task_id, **options)
        merged_options = dict(self.options, **options) if options else self.options
        option_task_id = merged_options.pop('task_id', None)
        if task_id is None:
            task_id = option_task_id
        return self.run(tasks, body, args, task_id=task_id, kwargs=kwargs, **merged_options)

    def apply(self, args=None, kwargs=None, propagate=True, body=None, **options):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        body = self.body if body is None else body
        tasks = self.tasks.clone() if isinstance(self.tasks, group) else group(self.tasks, app=self.app)
        return body.apply(args=(tasks.apply(args, kwargs).get(propagate=propagate),))

    @classmethod
    def _descend(cls, sig_obj):
        """Count the number of tasks in the given signature recursively.

        Descend into the signature object and return the amount of tasks it contains.
        """
        if not isinstance(sig_obj, Signature) and isinstance(sig_obj, dict):
            sig_obj = Signature.from_dict(sig_obj)
        if isinstance(sig_obj, group):
            subtasks = getattr(sig_obj.tasks, 'tasks', sig_obj.tasks)
            return sum((cls._descend(task) for task in subtasks))
        elif isinstance(sig_obj, _chain):
            for child_sig in sig_obj.tasks[-1::-1]:
                child_size = cls._descend(child_sig)
                if child_size > 0:
                    return child_size
            return 0
        elif isinstance(sig_obj, chord):
            return cls._descend(sig_obj.body)
        elif isinstance(sig_obj, Signature):
            return 1
        return len(sig_obj)

    def __length_hint__(self):
        """Return the number of tasks in this chord's header (recursively)."""
        tasks = getattr(self.tasks, 'tasks', self.tasks)
        return sum((self._descend(task) for task in tasks))

    def run(self, header, body, partial_args, app=None, interval=None, countdown=1, max_retries=None, eager=False, task_id=None, kwargs=None, **options):
        """Execute the chord.

        Executing the chord means executing the header and sending the
        result to the body. In case of an empty header, the body is
        executed immediately.

        Arguments:
            header (group): The header to execute.
            body (Signature): The body to execute.
            partial_args (tuple): Arguments to pass to the header.
            app (Celery): The Celery app instance.
            interval (float): The interval between retries.
            countdown (int): The countdown between retries.
            max_retries (int): The maximum number of retries.
            task_id (str): The task id to use for the body.
            kwargs (dict): Keyword arguments to pass to the header.
            options (dict): Options to pass to the header.

        Returns:
            AsyncResult: The result of the body (with the result of the header in the parent of the body).
        """
        app = app or self._get_app(body)
        group_id = header.options.get('task_id') or uuid()
        root_id = body.options.get('root_id')
        options = dict(self.options, **options) if options else self.options
        if options:
            options.pop('task_id', None)
            body.options.update(options)
        bodyres = body.freeze(task_id, root_id=root_id)
        options.pop('chain', None)
        options.pop('chord', None)
        options.pop('task_id', None)
        header_result_args = header._freeze_group_tasks(group_id=group_id, chord=body, root_id=root_id)
        if header.tasks:
            app.backend.apply_chord(header_result_args, body, interval=interval, countdown=countdown, max_retries=max_retries)
            header_result = header.apply_async(partial_args, kwargs, task_id=group_id, **options)
        else:
            body.delay([])
            header_result = self.app.GroupResult(*header_result_args)
        bodyres.parent = header_result
        return bodyres

    def clone(self, *args, **kwargs):
        signature = super().clone(*args, **kwargs)
        try:
            signature.kwargs['body'] = maybe_signature(signature.kwargs['body'], clone=True)
        except (AttributeError, KeyError):
            pass
        return signature

    def link(self, callback):
        """Links a callback to the chord body only."""
        self.body.link(callback)
        return callback

    def link_error(self, errback):
        """Links an error callback to the chord body, and potentially the header as well.

        Note:
            The ``task_allow_error_cb_on_chord_header`` setting controls whether
            error callbacks are allowed on the header. If this setting is
            ``False`` (the current default), then the error callback will only be
            applied to the body.
        """
        errback = maybe_signature(errback)
        if self.app.conf.task_allow_error_cb_on_chord_header:
            for task in maybe_list(self.tasks) or []:
                task.link_error(errback.clone(immutable=True))
        else:
            warnings.warn('task_allow_error_cb_on_chord_header=False is pending deprecation in a future release of Celery.\nPlease test the new behavior by setting task_allow_error_cb_on_chord_header to True and report any concerns you might have in our issue tracker before we make a final decision regarding how errbacks should behave when used with chords.', CPendingDeprecationWarning)
        self.body.link_error(errback)
        return errback

    def set_immutable(self, immutable):
        """Sets the immutable flag on the chord header only.

        Note:
            Does not affect the chord body.

        Arguments:
            immutable (bool): The new mutability value for chord header.
        """
        for task in self.tasks:
            task.set_immutable(immutable)

    def __repr__(self):
        if self.body:
            if isinstance(self.body, _chain):
                return remove_repeating_from_task(self.body.tasks[0]['task'], '%({} | {!r})'.format(self.body.tasks[0].reprcall(self.tasks), chain(self.body.tasks[1:], app=self._app)))
            return '%' + remove_repeating_from_task(self.body['task'], self.body.reprcall(self.tasks))
        return f'<chord without body: {self.tasks!r}>'

    @cached_property
    def app(self):
        return self._get_app(self.body)

    def _get_app(self, body=None):
        app = self._app
        if app is None:
            try:
                tasks = self.tasks.tasks
            except AttributeError:
                tasks = self.tasks
            if tasks:
                app = tasks[0]._app
            if app is None and body is not None:
                app = body._app
        return app if app is not None else current_app
    tasks = getitem_property('kwargs.header', 'Tasks in chord header.')
    body = getitem_property('kwargs.body', 'Body task of chord.')