import enum
import collections
import threading
import typing
from typing import Deque, Iterable, Sequence
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher import futures
from google.cloud.pubsub_v1.publisher import exceptions
from google.cloud.pubsub_v1.publisher._sequencer import base as sequencer_base
from google.cloud.pubsub_v1.publisher._batch import base as batch_base
from google.pubsub_v1 import types as gapic_types
def _set_batches(self, batches: Iterable['_batch.thread.Batch']) -> None:
    self._ordered_batches = collections.deque(batches)