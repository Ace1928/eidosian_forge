import functools
import hashlib
import multiprocessing.dummy
import os
import pathlib
import queue
import random
import shutil
import tarfile
import threading
import time
import typing
import urllib
import warnings
import weakref
import zipfile
from abc import abstractmethod
from contextlib import closing
import numpy as np
import tensorflow.compat.v2 as tf
from six.moves.urllib.parse import urlsplit
from keras.src.utils import io_utils
from keras.src.utils import tf_inspect
from keras.src.utils.generic_utils import Progbar
from tensorflow.python.util.tf_export import keras_export
from six.moves.urllib.request import urlopen
def _filter_safe_paths(members):
    base_dir = _resolve_path('.')
    for finfo in members:
        valid_path = False
        if _is_path_in_dir(finfo.name, base_dir):
            valid_path = True
            yield finfo
        elif finfo.issym() or finfo.islnk():
            if _is_link_in_dir(finfo, base_dir):
                valid_path = True
                yield finfo
        if not valid_path:
            warnings.warn(f"Skipping invalid path during archive extraction: '{finfo.name}'.")