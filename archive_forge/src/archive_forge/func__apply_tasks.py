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
def _apply_tasks(self, tasks, producer=None, app=None, p=None, add_to_parent=None, chord=None, args=None, kwargs=None, **options):
    """Run all the tasks in the group.

        This is used by :meth:`apply_async` to run all the tasks in the group
        and return a generator of their results.

        Arguments:
            tasks (list): List of tasks in the group.
            producer (Producer): The producer to use to publish the tasks.
            app (Celery): The Celery app instance.
            p (barrier): Barrier object to synchronize the tasks results.
            args (list): List of arguments to be prepended to
                the arguments of each task.
            kwargs (dict): Dict of keyword arguments to be merged with
                the keyword arguments of each task.
            **options (dict): Options to be merged with the options of each task.

        Returns:
            generator: A generator for the AsyncResult of the tasks in the group.
        """
    app = app or self.app
    with app.producer_or_acquire(producer) as producer:
        chord_size = 0
        tasks_shifted, tasks = itertools.tee(tasks)
        next(tasks_shifted, None)
        next_task = next(tasks_shifted, None)
        for task_index, current_task in enumerate(tasks):
            sig, res, group_id = current_task
            chord_obj = chord if chord is not None else sig.options.get('chord')
            chord_size += _chord._descend(sig)
            if chord_obj is not None and next_task is None:
                app.backend.set_chord_size(group_id, chord_size)
            sig.apply_async(producer=producer, add_to_parent=False, chord=chord_obj, args=args, kwargs=kwargs, **options)
            if p and (not p.cancelled) and (not p.ready):
                p.size += 1
                res.then(p, weak=True)
            next_task = next(tasks_shifted, None)
            yield res