import os
import platform
import signal
import time
import warnings
from _thread import interrupt_main  # Py 3
from threading import Thread
from traitlets.log import get_logger
Run the poll loop. This method never returns.