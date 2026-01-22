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
def _load_ipython_extension(ipython):
    """Load the TensorBoard notebook extension.

    Intended to be called from `%load_ext tensorboard`. Do not invoke this
    directly.

    Args:
      ipython: An `IPython.InteractiveShell` instance.
    """
    _register_magics(ipython)