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
@pytest.mark.parametrize('ctx', [_get_temp_local_path, _get_temp_gcs_path, _get_temp_as_path])
def test_set_mtime(ctx):
    contents = b'meow!'
    with ctx() as path:
        _write_contents(path, contents)
        s = bf.stat(path)
        assert abs(time.time() - s.mtime) <= 30
        new_mtime = 1
        assert bf.set_mtime(path, new_mtime)
        assert bf.stat(path).mtime == new_mtime