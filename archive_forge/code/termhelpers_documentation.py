import tty
import termios
import fcntl
import os
from typing import IO, ContextManager, Type, List, Union, Optional
from types import TracebackType

    A context manager for making an input stream nonblocking.
    