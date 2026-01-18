import typing
from typing import Optional
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher._sequencer import base
from google.pubsub_v1 import types as gapic_types
Batch message into existing or new batch.

        Args:
            message:
                The Pub/Sub message.
            retry:
                The retry settings to apply when publishing the message.
            timeout:
                The timeout to apply when publishing the message.

        Returns:
            An object conforming to the :class:`~concurrent.futures.Future` interface.
            The future tracks the publishing status of the message.

        Raises:
            RuntimeError:
                If called after stop() has already been called.

            pubsub_v1.publisher.exceptions.MessageTooLargeError: If publishing
                the ``message`` would exceed the max size limit on the backend.
        