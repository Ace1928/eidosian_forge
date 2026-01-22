import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
bool: True if the response stream is paused.