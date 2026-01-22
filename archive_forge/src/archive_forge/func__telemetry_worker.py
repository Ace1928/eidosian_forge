from queue import Queue
from threading import Lock, Thread
from typing import Dict, Optional, Union
from urllib.parse import quote
from .. import constants, logging
from . import build_hf_headers, get_session, hf_raise_for_status
def _telemetry_worker():
    """Wait for a task and consume it."""
    while True:
        kwargs = _TELEMETRY_QUEUE.get()
        _send_telemetry_in_thread(**kwargs)
        _TELEMETRY_QUEUE.task_done()