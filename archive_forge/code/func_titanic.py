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
def titanic():
    url = 'https://storage.mds.yandex.net/get-devtools-opensource/233854/titanic.tar.gz'
    md5 = '9c8bc61d545c6af244a1d37494df3fc3'
    dataset_name, train_file, test_file = ('titanic', 'train.csv', 'test.csv')
    return _load_dataset_pd(url, md5, dataset_name, train_file, test_file)