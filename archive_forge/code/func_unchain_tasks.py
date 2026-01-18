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