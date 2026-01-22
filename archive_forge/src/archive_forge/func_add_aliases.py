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
@param.parameterized.bothmethod
def add_aliases(self_or_cls, **kwargs):
    """
        Conveniently add new aliases as keyword arguments. For instance
        you can add a new alias with add_aliases(short='Longer string')
        """
    self_or_cls.aliases.update({v: k for k, v in kwargs.items()})