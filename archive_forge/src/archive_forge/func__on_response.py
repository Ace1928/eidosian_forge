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
def _on_response(self, response: gapic_types.StreamingPullResponse) -> None:
    """Process all received Pub/Sub messages.

        For each message, send a modified acknowledgment request to the
        server. This prevents expiration of the message due to buffering by
        gRPC or proxy/firewall. This makes the server and client expiration
        timer closer to each other thus preventing the message being
        redelivered multiple times.

        After the messages have all had their ack deadline updated, execute
        the callback for each message using the executor.
        """
    if response is None:
        _LOGGER.debug('Response callback invoked with None, likely due to a transport shutdown.')
        return
    received_messages = response._pb.received_messages
    _LOGGER.debug('Processing %s received message(s), currently on hold %s (bytes %s).', len(received_messages), self._messages_on_hold.size, self._on_hold_bytes)
    with self._exactly_once_enabled_lock:
        if response.subscription_properties.exactly_once_delivery_enabled != self._exactly_once_enabled:
            self._exactly_once_enabled = response.subscription_properties.exactly_once_delivery_enabled
            self._obtain_ack_deadline(maybe_update=True)
            self._send_new_ack_deadline = True
    ack_id_gen = (message.ack_id for message in received_messages)
    expired_ack_ids = self._send_lease_modacks(ack_id_gen, self.ack_deadline, warn_on_invalid=False)
    with self._pause_resume_lock:
        assert self._scheduler is not None
        assert self._leaser is not None
        for received_message in received_messages:
            if not self._exactly_once_delivery_enabled() or received_message.ack_id not in expired_ack_ids:
                message = google.cloud.pubsub_v1.subscriber.message.Message(received_message.message, received_message.ack_id, received_message.delivery_attempt, self._scheduler.queue, self._exactly_once_delivery_enabled)
                self._messages_on_hold.put(message)
                self._on_hold_bytes += message.size
                req = requests.LeaseRequest(ack_id=message.ack_id, byte_size=message.size, ordering_key=message.ordering_key)
                self._leaser.add([req])
        self._maybe_release_messages()
    self.maybe_pause_consumer()