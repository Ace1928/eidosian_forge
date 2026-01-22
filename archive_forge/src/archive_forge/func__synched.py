from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def _synched(func):

    @wraps(func)
    def _synchronizer(self, *args, **kwargs):
        tlock = getattr(self, tlockname)
        logger.debug('acquiring lock %r for %s', tlockname, func.__name__)
        with tlock:
            logger.debug('acquired lock %r for %s', tlockname, func.__name__)
            result = func(self, *args, **kwargs)
            logger.debug('releasing lock %r for %s', tlockname, func.__name__)
            return result
    return _synchronizer