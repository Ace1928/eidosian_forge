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
class _OrderedSequencerStatus(str, enum.Enum):
    """An enum-like class representing valid statuses for an OrderedSequencer.

    Starting state: ACCEPTING_MESSAGES
    Valid transitions:
      ACCEPTING_MESSAGES -> PAUSED (on permanent error)
      ACCEPTING_MESSAGES -> STOPPED  (when user calls stop() explicitly)
      ACCEPTING_MESSAGES -> FINISHED  (all batch publishes finish normally)

      PAUSED -> ACCEPTING_MESSAGES  (when user unpauses)
      PAUSED -> STOPPED  (when user calls stop() explicitly)

      STOPPED -> FINISHED (user stops client and the one remaining batch finishes
                           publish)
      STOPPED -> PAUSED (stop() commits one batch, which fails permanently)

      FINISHED -> ACCEPTING_MESSAGES (publish happens while waiting for cleanup)
      FINISHED -> STOPPED (when user calls stop() explicitly)
    Illegal transitions:
      PAUSED -> FINISHED (since all batches are cancelled on pause, there should
                          not be any that finish normally. paused sequencers
                          should not be cleaned up because their presence
                          indicates that the ordering key needs to be resumed)
      STOPPED -> ACCEPTING_MESSAGES (no way to make a user-stopped sequencer
                                     accept messages again. this is okay since
                                     stop() should only be called on shutdown.)
      FINISHED -> PAUSED (no messages remain in flight, so they can't cause a
                          permanent error and pause the sequencer)
    """
    ACCEPTING_MESSAGES = 'accepting messages'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    FINISHED = 'finished'