import codecs
import logging
import os
import pickle
import re
import shutil
import tarfile
from contextlib import suppress
import joblib
import numpy as np
import scipy.sparse as sp
from .. import preprocessing
from ..feature_extraction.text import CountVectorizer
from ..utils import Bunch, check_random_state
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home, load_files
from ._base import (
def _download_20newsgroups(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)
    os.makedirs(target_dir, exist_ok=True)
    logger.info('Downloading dataset from %s (14 MB)', ARCHIVE.url)
    archive_path = _fetch_remote(ARCHIVE, dirname=target_dir)
    logger.debug('Decompressing %s', archive_path)
    tarfile.open(archive_path, 'r:gz').extractall(path=target_dir)
    with suppress(FileNotFoundError):
        os.remove(archive_path)
    cache = dict(train=load_files(train_path, encoding='latin1'), test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)
    shutil.rmtree(target_dir)
    return cache