import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
class RejectedSubmission(Exception):
    """Exception raised when a submitted call is rejected (for some reason)."""