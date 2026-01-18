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
@pytest.mark.slow
@pytest.mark.parametrize('ctx', [_get_temp_gcs_path, _get_temp_as_path])
def test_parallel_copy_large_file(ctx):
    contents = b'meow!' * common.PARALLEL_COPY_MINIMUM_PART_SIZE + b'meow???'
    with ctx() as remote_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, 'test.txt')
            with open(local_path, 'wb') as f:
                f.write(contents)
            bf.copy(local_path, remote_path, parallel=True)
        assert _read_contents(remote_path) == contents
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, 'test.txt')
            bf.copy(remote_path, local_path, parallel=True)
            assert _read_contents(local_path) == contents