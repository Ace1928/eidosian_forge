import multiprocessing
import os
import re
import sys
import time
from .processes import ForkedProcess
from .remoteproxy import ClosedError
class CanceledError(Exception):
    """Raised when the progress dialog is canceled during a processing operation."""
    pass