from __future__ import unicode_literals
import fcntl
import os
import random
import signal
import threading
import time
from prompt_toolkit.terminal.vt100_input import InputStream
from prompt_toolkit.utils import DummyContext, in_main_thread
from prompt_toolkit.input import Input
from .base import EventLoop, INPUT_TIMEOUT
from .callbacks import EventLoopCallbacks
from .inputhook import InputHookContext
from .posix_utils import PosixStdinReader
from .utils import TimeIt
from .select import AutoSelector, Selector, fd_to_int
 True when there is input ready. The inputhook should return control. 