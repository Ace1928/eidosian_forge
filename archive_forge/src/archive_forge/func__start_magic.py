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
def _start_magic(line):
    """Implementation of the `%tensorboard` line magic."""
    return start(line)