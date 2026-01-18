import datetime
import warnings
import random
import string
import tempfile
import os
import contextlib
import json
import urllib.request
import hashlib
import time
import subprocess as sp
import multiprocessing as mp
import platform
import pickle
import zipfile
import re
import av
import pytest
from tensorflow.io import gfile
import imageio
import numpy as np
import blobfile as bf
from blobfile import _ops as ops, _azure as azure, _common as common
def test_windowed_file():
    with _get_temp_local_path() as path:
        with open(path, 'wb') as f:
            f.write(b'meow')
        with open(path, 'rb') as f:
            f2 = common.WindowedFile(f, start=1, end=3)
            assert f2.read() == b'eo'
            f2.seek(0)
            assert f2.read(1) + f2.read(1) + f2.read(1) == b'eo'
            with pytest.raises(AssertionError):
                f2.seek(-1)
            with pytest.raises(AssertionError):
                f2.seek(2)