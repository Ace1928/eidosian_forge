import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def _register_magics(ipython):
    """Register IPython line/cell magics.

    Args:
      ipython: An `InteractiveShell` instance.
    """
    ipython.register_magic_function(_start_magic, magic_kind='line', magic_name='tensorboard')