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
def test_scanglob(ctx):
    contents = b'meow!'
    with ctx() as path:
        dirpath = bf.dirname(path)
        a_path = bf.join(dirpath, 'ab')
        with bf.BlobFile(a_path, 'wb') as w:
            w.write(contents)
        b_path = bf.join(dirpath, 'bb')
        with bf.BlobFile(b_path, 'wb') as w:
            w.write(contents)
        path = bf.join(dirpath, 'test.txt')
        with bf.BlobFile(path, 'wb') as w:
            w.write(contents)
        path = bf.join(dirpath, 'subdir', 'test.txt')
        bf.makedirs(bf.dirname(path))
        with bf.BlobFile(path, 'wb') as f:
            f.write(contents)
        entries = sorted(list(bf.scanglob(bf.join(dirpath, '*b*'))))
        assert entries[0].name == 'ab' and entries[0].is_file
        assert entries[1].name == 'bb' and entries[1].is_file
        assert entries[2].name == 'subdir' and entries[2].is_dir
        for shard_prefix_length in [0, 1]:
            for pattern in ['*b', 'b*', '**', 'b**', '**t', '*b*']:
                normal_entries = sorted(list(bf.scanglob(bf.join(dirpath, pattern))))
                parallel_entries = sorted(list(bf.scanglob(bf.join(dirpath, pattern), parallel=True, shard_prefix_length=shard_prefix_length)))
                assert parallel_entries == normal_entries
        if '://' in path:
            entries = sorted(list(bf.scanglob(bf.join(dirpath, '**'))))
            assert entries[0].name == 'ab' and entries[0].is_file
            assert entries[1].name == 'bb' and entries[1].is_file
            assert entries[2].name == 'subdir' and entries[2].is_dir