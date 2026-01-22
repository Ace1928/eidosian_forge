import errno
import logging
import os
from gzip import GzipFile
from os.path import exists, join
import joblib
import numpy as np
from ..utils import Bunch, check_random_state
from ..utils import shuffle as shuffle_method
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home
from ._base import (
def _fetch_brute_kddcup99(data_home=None, download_if_missing=True, percent10=True):
    """Load the kddcup99 dataset, downloading it if necessary.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (494021, 41)
            Each row corresponds to the 41 features in the dataset.
        target : ndarray of shape (494021,)
            Each value corresponds to one of the 21 attack types or to the
            label 'normal.'.
        feature_names : list
            The names of the dataset columns
        target_names: list
            The names of the target columns
        DESCR : str
            Description of the kddcup99 dataset.

    """
    data_home = get_data_home(data_home=data_home)
    dir_suffix = '-py3'
    if percent10:
        kddcup_dir = join(data_home, 'kddcup99_10' + dir_suffix)
        archive = ARCHIVE_10_PERCENT
    else:
        kddcup_dir = join(data_home, 'kddcup99' + dir_suffix)
        archive = ARCHIVE
    samples_path = join(kddcup_dir, 'samples')
    targets_path = join(kddcup_dir, 'targets')
    available = exists(samples_path)
    dt = [('duration', int), ('protocol_type', 'S4'), ('service', 'S11'), ('flag', 'S6'), ('src_bytes', int), ('dst_bytes', int), ('land', int), ('wrong_fragment', int), ('urgent', int), ('hot', int), ('num_failed_logins', int), ('logged_in', int), ('num_compromised', int), ('root_shell', int), ('su_attempted', int), ('num_root', int), ('num_file_creations', int), ('num_shells', int), ('num_access_files', int), ('num_outbound_cmds', int), ('is_host_login', int), ('is_guest_login', int), ('count', int), ('srv_count', int), ('serror_rate', float), ('srv_serror_rate', float), ('rerror_rate', float), ('srv_rerror_rate', float), ('same_srv_rate', float), ('diff_srv_rate', float), ('srv_diff_host_rate', float), ('dst_host_count', int), ('dst_host_srv_count', int), ('dst_host_same_srv_rate', float), ('dst_host_diff_srv_rate', float), ('dst_host_same_src_port_rate', float), ('dst_host_srv_diff_host_rate', float), ('dst_host_serror_rate', float), ('dst_host_srv_serror_rate', float), ('dst_host_rerror_rate', float), ('dst_host_srv_rerror_rate', float), ('labels', 'S16')]
    column_names = [c[0] for c in dt]
    target_names = column_names[-1]
    feature_names = column_names[:-1]
    if available:
        try:
            X = joblib.load(samples_path)
            y = joblib.load(targets_path)
        except Exception as e:
            raise OSError(f'The cache for fetch_kddcup99 is invalid, please delete {str(kddcup_dir)} and run the fetch_kddcup99 again') from e
    elif download_if_missing:
        _mkdirp(kddcup_dir)
        logger.info('Downloading %s' % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir)
        DT = np.dtype(dt)
        logger.debug('extracting archive')
        archive_path = join(kddcup_dir, archive.filename)
        file_ = GzipFile(filename=archive_path, mode='r')
        Xy = []
        for line in file_.readlines():
            line = line.decode()
            Xy.append(line.replace('\n', '').split(','))
        file_.close()
        logger.debug('extraction done')
        os.remove(archive_path)
        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])
        X = Xy[:, :-1]
        y = Xy[:, -1]
        joblib.dump(X, samples_path, compress=0)
        joblib.dump(y, targets_path, compress=0)
    else:
        raise OSError('Data not found and `download_if_missing` is False')
    return Bunch(data=X, target=y, feature_names=feature_names, target_names=[target_names])