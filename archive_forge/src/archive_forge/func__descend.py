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