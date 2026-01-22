import logging
from numbers import Integral, Real
from os import PathLike, listdir, makedirs, remove
from os.path import exists, isdir, join
import numpy as np
from joblib import Memory
from ..utils import Bunch
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ._base import (
def _check_fetch_lfw(data_home=None, funneled=True, download_if_missing=True):
    """Helper function to download any missing LFW data"""
    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, 'lfw_home')
    if not exists(lfw_home):
        makedirs(lfw_home)
    for target in TARGETS:
        target_filepath = join(lfw_home, target.filename)
        if not exists(target_filepath):
            if download_if_missing:
                logger.info('Downloading LFW metadata: %s', target.url)
                _fetch_remote(target, dirname=lfw_home)
            else:
                raise OSError('%s is missing' % target_filepath)
    if funneled:
        data_folder_path = join(lfw_home, 'lfw_funneled')
        archive = FUNNELED_ARCHIVE
    else:
        data_folder_path = join(lfw_home, 'lfw')
        archive = ARCHIVE
    if not exists(data_folder_path):
        archive_path = join(lfw_home, archive.filename)
        if not exists(archive_path):
            if download_if_missing:
                logger.info('Downloading LFW data (~200MB): %s', archive.url)
                _fetch_remote(archive, dirname=lfw_home)
            else:
                raise OSError('%s is missing' % archive_path)
        import tarfile
        logger.debug('Decompressing the data archive to %s', data_folder_path)
        tarfile.open(archive_path, 'r:gz').extractall(path=lfw_home)
        remove(archive_path)
    return (lfw_home, data_folder_path)