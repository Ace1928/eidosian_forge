import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def _process_underscores(self, tokens):
    """Strip underscores to make sure the number is correct after join"""
    groups = [[str(''.join(el))] if b else list(el) for b, el in itertools.groupby(tokens, lambda k: k == '_')]
    flattened = [el for group in groups for el in group]
    processed = []
    for token in flattened:
        if token == '_':
            continue
        if token.startswith('_'):
            token = str(token[1:])
        if token.endswith('_'):
            token = str(token[:-1])
        processed.append(token)
    return processed