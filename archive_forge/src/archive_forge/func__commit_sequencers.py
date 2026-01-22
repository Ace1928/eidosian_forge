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
def _commit_sequencers(self) -> None:
    """Clean up finished sequencers and commit the rest."""
    finished_sequencer_keys = [key for key, sequencer in self._sequencers.items() if sequencer.is_finished()]
    for sequencer_key in finished_sequencer_keys:
        del self._sequencers[sequencer_key]
    for sequencer in self._sequencers.values():
        sequencer.commit()