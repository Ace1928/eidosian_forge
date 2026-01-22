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
class StampingVisitor(metaclass=ABCMeta):
    """Stamping API.  A class that provides a stamping API possibility for
    canvas primitives. If you want to implement stamping behavior for
    a canvas primitive override method that represents it.
    """

    def on_group_start(self, group, **headers) -> dict:
        """Method that is called on group stamping start.

         Arguments:
             group (group): Group that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """
        return {}

    def on_group_end(self, group, **headers) -> None:
        """Method that is called on group stamping end.

         Arguments:
             group (group): Group that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         """
        pass

    def on_chain_start(self, chain, **headers) -> dict:
        """Method that is called on chain stamping start.

         Arguments:
             chain (chain): Chain that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """
        return {}

    def on_chain_end(self, chain, **headers) -> None:
        """Method that is called on chain stamping end.

         Arguments:
             chain (chain): Chain that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         """
        pass

    @abstractmethod
    def on_signature(self, sig, **headers) -> dict:
        """Method that is called on signature stamping.

         Arguments:
             sig (Signature): Signature that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """

    def on_chord_header_start(self, sig, **header) -> dict:
        """Method that is called on сhord header stamping start.

         Arguments:
             sig (chord): chord that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """
        if not isinstance(sig.tasks, group):
            sig.tasks = group(sig.tasks)
        return self.on_group_start(sig.tasks, **header)

    def on_chord_header_end(self, sig, **header) -> None:
        """Method that is called on сhord header stamping end.

           Arguments:
               sig (chord): chord that is stamped.
               headers (Dict): Partial headers that could be merged with existing headers.
        """
        self.on_group_end(sig.tasks, **header)

    def on_chord_body(self, sig, **header) -> dict:
        """Method that is called on chord body stamping.

         Arguments:
             sig (chord): chord that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
        """
        return {}

    def on_callback(self, callback, **header) -> dict:
        """Method that is called on callback stamping.

         Arguments:
             callback (Signature): callback that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """
        return {}

    def on_errback(self, errback, **header) -> dict:
        """Method that is called on errback stamping.

         Arguments:
             errback (Signature): errback that is stamped.
             headers (Dict): Partial headers that could be merged with existing headers.
         Returns:
             Dict: headers to update.
         """
        return {}