from __future__ import absolute_import
import copy
import logging
import os
import threading
import time
import typing
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import warnings
from google.api_core import gapic_v1
from google.auth.credentials import AnonymousCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
from google.cloud.pubsub_v1.publisher import futures
from google.cloud.pubsub_v1.publisher._batch import thread
from google.cloud.pubsub_v1.publisher._sequencer import ordered_sequencer
from google.cloud.pubsub_v1.publisher._sequencer import unordered_sequencer
from google.cloud.pubsub_v1.publisher.flow_controller import FlowController
from google.pubsub_v1 import gapic_version as package_version
from google.pubsub_v1 import types as gapic_types
from google.pubsub_v1.services.publisher import client as publisher_client
def _get_or_create_sequencer(self, topic: str, ordering_key: str) -> SequencerType:
    """Get an existing sequencer or create a new one given the (topic,
        ordering_key) pair.
        """
    sequencer_key = (topic, ordering_key)
    sequencer = self._sequencers.get(sequencer_key)
    if sequencer is None:
        if ordering_key == '':
            sequencer = unordered_sequencer.UnorderedSequencer(self, topic)
        else:
            sequencer = ordered_sequencer.OrderedSequencer(self, topic, ordering_key)
        self._sequencers[sequencer_key] = sequencer
    return sequencer