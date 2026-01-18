from __future__ import absolute_import
import copy
import logging
import random
import threading
import time
import typing
from typing import Dict, Iterable, Optional, Union
from google.cloud.pubsub_v1.subscriber._protocol.dispatcher import _MAX_BATCH_LATENCY
from google.cloud.pubsub_v1.subscriber._protocol import requests
Maintain all of the leases being managed.

        This method modifies the ack deadline for all of the managed
        ack IDs, then waits for most of that time (but with jitter), and
        repeats.
        