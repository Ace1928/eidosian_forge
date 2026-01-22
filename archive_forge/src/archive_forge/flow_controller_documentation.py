from collections import OrderedDict
import logging
import threading
from typing import Dict, Optional, Type
import warnings
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
Return the current flow control load information.

        The caller can optionally adjust some of the values to fit its reporting
        needs.

        The method assumes that the caller has obtained ``_operational_lock``.

        Args:
            message_count:
                The value to override the current message count with.
            total_bytes:
                The value to override the current total bytes with.
        