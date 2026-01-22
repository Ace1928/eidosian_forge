import collections
import contextlib
import cProfile
from io import StringIO
import gc
import os
import multiprocessing
import sys
import time
import unittest
import warnings
from unittest import result, runner, signals

        A context manager which cleans up unwanted attributes on a test case
        (or any other object).
        