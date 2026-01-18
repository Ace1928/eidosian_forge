import os
import signal
import socket
import sys
import threading
from . import process
from .context import reduction
from . import util
Get the fd.  This should only be called once.