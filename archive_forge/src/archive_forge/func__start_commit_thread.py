from __future__ import absolute_import
import logging
import threading
import time
import typing
from typing import Any, Callable, List, Optional, Sequence
import google.api_core.exceptions
from google.api_core import gapic_v1
from google.cloud.pubsub_v1.publisher import exceptions
from google.cloud.pubsub_v1.publisher import futures
from google.cloud.pubsub_v1.publisher._batch import base
from google.pubsub_v1 import types as gapic_types
def _start_commit_thread(self) -> None:
    """Start a new thread to actually handle the commit."""
    commit_thread = threading.Thread(name='Thread-CommitBatchPublisher', target=self._commit, daemon=True)
    commit_thread.start()