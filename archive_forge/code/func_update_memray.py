import io
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from ..config import config
from ..util import escape
from .state import state
def update_memray(*args):
    update_profiles(reporter=reporter.value)