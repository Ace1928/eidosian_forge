from __future__ import unicode_literals
import functools
import os
import signal
import six
import sys
import textwrap
import threading
import time
import types
import weakref
from subprocess import Popen
from .application import Application, AbortAction
from .buffer import Buffer
from .buffer_mapping import BufferMapping
from .completion import CompleteEvent, get_common_complete_suffix
from .enums import SEARCH_BUFFER
from .eventloop.base import EventLoop
from .eventloop.callbacks import EventLoopCallbacks
from .filters import Condition
from .input import StdinInput, Input
from .key_binding.input_processor import InputProcessor
from .key_binding.input_processor import KeyPress
from .key_binding.registry import Registry
from .key_binding.vi_state import ViState
from .keys import Keys
from .output import Output
from .renderer import Renderer, print_tokens
from .search_state import SearchState
from .utils import Event
from .buffer import AcceptAction
def wait_for_enter():
    """
            Create a sub application to wait for the enter key press.
            This has two advantages over using 'input'/'raw_input':
            - This will share the same input/output I/O.
            - This doesn't block the event loop.
            """
    from .shortcuts import create_prompt_application
    registry = Registry()

    @registry.add_binding(Keys.ControlJ)
    @registry.add_binding(Keys.ControlM)
    def _(event):
        event.cli.set_return_value(None)
    application = create_prompt_application(message='Press ENTER to continue...', key_bindings_registry=registry)
    self.run_sub_application(application)