import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def _load_dataset_pd(url, md5, dataset_name, train_file, test_file, sep=',', header='infer', cache=False):
    train_path, test_path = _download_dataset(url, md5, dataset_name, train_file, test_file, cache)
    train, test = (pd.read_csv(train_path, header=header, sep=sep), pd.read_csv(test_path, header=header, sep=sep))
    if not cache:
        os.remove(train_path)
        os.remove(test_path)
    return (train, test)