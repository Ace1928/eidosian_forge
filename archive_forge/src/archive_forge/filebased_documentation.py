import glob
import os
import pickle
import random
import tempfile
import time
import zlib
from hashlib import md5
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.core.files import locks
from django.core.files.move import file_move_safe

        Get a list of paths to all the cache files. These are all the files
        in the root cache dir that end on the cache_suffix.
        