import os
import zstandard
import ujson as json
import time
import tarfile
import codecs
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path
def _listdir_or_file(x):
    if isinstance(x, list):
        return reduce(lambda x, y: x + y, map(listdir_or_file, sorted(x)))
    if os.path.isfile(x):
        return [x]
    elif os.path.isdir(x):
        return [str(Path(x) / fn) for fn in sorted(os.listdir(x))]
    else:
        raise FileNotFoundError(f'{x} not found')