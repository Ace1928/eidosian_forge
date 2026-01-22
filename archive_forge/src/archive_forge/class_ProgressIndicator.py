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
class ProgressIndicator(param.Parameterized):
    """
    Baseclass for any ProgressIndicator that indicates progress
    as a completion percentage.
    """
    percent_range = param.NumericTuple(default=(0.0, 100.0), doc='\n        The total percentage spanned by the progress bar when called\n        with a value between 0% and 100%. This allows an overall\n        completion in percent to be broken down into smaller sub-tasks\n        that individually complete to 100 percent.')
    label = param.String(default='Progress', allow_None=True, doc='\n        The label of the current progress bar.')

    def __call__(self, completion):
        raise NotImplementedError