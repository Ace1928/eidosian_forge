from __future__ import absolute_import
from __future__ import division
import functools
import itertools
import logging
import math
import time
import threading
import typing
from typing import List, Optional, Sequence, Union
import warnings
from google.api_core.retry import exponential_sleep_generator
from google.cloud.pubsub_v1.subscriber._protocol import helper_threads
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
def _start_retry_thread(self, thread_name, thread_target):
    retry_thread = threading.Thread(name=thread_name, target=thread_target, daemon=True)
    retry_thread.start()