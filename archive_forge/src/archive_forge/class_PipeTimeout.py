import array
import threading
import time
from paramiko.util import b
class PipeTimeout(IOError):
    """
    Indicates that a timeout was reached on a read from a `.BufferedPipe`.
    """
    pass