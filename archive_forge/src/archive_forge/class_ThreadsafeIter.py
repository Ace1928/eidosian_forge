from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
class ThreadsafeIter(object):
    """Wrap an iterator with a lock and propagate exceptions to all threads."""

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        self._exception = None

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        with self.lock:
            if self._exception:
                raise self._exception
            try:
                return next(self.it)
            except Exception as e:
                self._exception = e
                raise