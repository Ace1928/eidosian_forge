import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
class StartTraceback(Exception):
    """These exceptions (and their tracebacks) can be skipped with `skip_exceptions`"""
    pass