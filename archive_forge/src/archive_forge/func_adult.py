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
def adult():
    """
    Download "Adult Data Set" [1] from UCI Machine Learning Repository.

    Will return two pandas.DataFrame-s, first with train part (adult.data) and second with test part
    (adult.test) of the dataset.

    [1]: https://archive.ics.uci.edu/ml/datasets/Adult
    """
    names = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income')
    dtype = {'age': float, 'workclass': object, 'fnlwgt': float, 'education': object, 'education-num': float, 'marital-status': object, 'occupation': object, 'relationship': object, 'race': object, 'sex': object, 'capital-gain': float, 'capital-loss': float, 'hours-per-week': float, 'native-country': object, 'income': object}
    train_urls = ('https://proxy.sandbox.yandex-team.ru/779118052', 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    train_md5 = '5d7c39d7b8804f071cdd1f2a7c460872'
    train_path = tempfile.mktemp()
    _cached_download(train_urls, train_md5, train_path)
    test_urls = ('https://proxy.sandbox.yandex-team.ru/779120000', 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')
    test_md5 = '35238206dfdf7f1fe215bbb874adecdc'
    test_path = tempfile.mktemp()
    _cached_download(test_urls, test_md5, test_path)
    train_df = pd.read_csv(train_path, names=names, header=None, sep=',\\s*', na_values=['?'], engine='python')
    os.remove(train_path)
    test_df = pd.read_csv(test_path, names=names, header=None, sep=',\\s*', na_values=['?'], skiprows=1, converters={'income': lambda x: x[:-1]}, engine='python')
    os.remove(test_path)
    train_df = train_df.astype(dtype)
    test_df = test_df.astype(dtype)
    return (train_df, test_df)