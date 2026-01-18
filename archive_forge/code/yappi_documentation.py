import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager

        Prints all of the thread profiler results to a given file. (stdout by default)
        