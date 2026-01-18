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
def test_overwrite_while_reading(ctx):
    chunk_size = 8 * 2 ** 20
    contents = b'\x00' * chunk_size * 2
    alternative_contents = b'\xff' * chunk_size * 4
    with ctx() as path:
        with bf.BlobFile(path, 'wb') as f:
            f.write(contents)
        with bf.BlobFile(path, 'rb') as f:
            read_contents = f.read(chunk_size)
            with bf.BlobFile(path, 'wb') as f2:
                f2.write(alternative_contents)
            f.raw._f = None
            read_contents += f.read(chunk_size)
            assert read_contents == contents[:chunk_size] + alternative_contents[chunk_size:chunk_size * 2]