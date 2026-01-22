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
def _get_initial_request(self, stream_ack_deadline_seconds: int) -> gapic_types.StreamingPullRequest:
    """Return the initial request for the RPC.

        This defines the initial request that must always be sent to Pub/Sub
        immediately upon opening the subscription.

        Args:
            stream_ack_deadline_seconds:
                The default message acknowledge deadline for the stream.

        Returns:
            A request suitable for being the first request on the stream (and not
            suitable for any other purpose).
        """
    request = gapic_types.StreamingPullRequest(stream_ack_deadline_seconds=stream_ack_deadline_seconds, modify_deadline_ack_ids=[], modify_deadline_seconds=[], subscription=self._subscription, client_id=self._client_id, max_outstanding_messages=0 if self._use_legacy_flow_control else self._flow_control.max_messages, max_outstanding_bytes=0 if self._use_legacy_flow_control else self._flow_control.max_bytes)
    return request