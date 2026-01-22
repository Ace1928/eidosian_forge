from __future__ import division
import collections
import functools
import itertools
import logging
import threading
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Set, Tuple
import uuid
import grpc  # type: ignore
from google.api_core import bidi
from google.api_core import exceptions
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.subscriber._protocol import dispatcher
from google.cloud.pubsub_v1.subscriber._protocol import heartbeater
from google.cloud.pubsub_v1.subscriber._protocol import histogram
from google.cloud.pubsub_v1.subscriber._protocol import leaser
from google.cloud.pubsub_v1.subscriber._protocol import messages_on_hold
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
import google.cloud.pubsub_v1.subscriber.message
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.pubsub_v1 import types as gapic_types
from google.rpc.error_details_pb2 import ErrorInfo  # type: ignore
from google.rpc import code_pb2  # type: ignore
from google.rpc import status_pb2
def activate_ordering_keys(self, ordering_keys: Iterable[str]) -> None:
    """Send the next message in the queue for each of the passed-in
        ordering keys, if they exist. Clean up state for keys that no longer
        have any queued messages.

        Since the load went down by one message, it's probably safe to send the
        user another message for the same key. Since the released message may be
        bigger than the previous one, this may increase the load above the maximum.
        This decision is by design because it simplifies MessagesOnHold.

        Args:
            ordering_keys:
                A sequence of ordering keys to activate. May be empty.
        """
    with self._pause_resume_lock:
        if self._scheduler is None:
            return
        self._messages_on_hold.activate_ordering_keys(ordering_keys, self._schedule_message_on_hold)