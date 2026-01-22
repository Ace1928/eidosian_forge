import ctypes
import signal
import threading
class BaseTimeoutException(Exception):
    """Base exception for timeouts."""
    pass