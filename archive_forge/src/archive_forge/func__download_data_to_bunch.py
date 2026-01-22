import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import (
from ..utils._param_validation import (
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
def _download_data_to_bunch(url: str, sparse: bool, data_home: Optional[str], *, as_frame: bool, openml_columns_info: List[dict], data_columns: List[str], target_columns: List[str], shape: Optional[Tuple[int, int]], md5_checksum: str, n_retries: int=3, delay: float=1.0, parser: str, read_csv_kwargs: Optional[Dict]=None):
    """Download ARFF data, load it to a specific container and create to Bunch.

    This function has a mechanism to retry/cache/clean the data.

    Parameters
    ----------
    url : str
        The URL of the ARFF file on OpenML.

    sparse : bool
        Whether the dataset is expected to use the sparse ARFF format.

    data_home : str
        The location where to cache the data.

    as_frame : bool
        Whether or not to return the data into a pandas DataFrame.

    openml_columns_info : list of dict
        The information regarding the columns provided by OpenML for the
        ARFF dataset. The information is stored as a list of dictionaries.

    data_columns : list of str
        The list of the features to be selected.

    target_columns : list of str
        The list of the target variables to be selected.

    shape : tuple or None
        With `parser="liac-arff"`, when using a generator to load the data,
        one needs to provide the shape of the data beforehand.

    md5_checksum : str
        The MD5 checksum provided by OpenML to check the data integrity.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    parser : {"liac-arff", "pandas"}
        The parser used to parse the ARFF file.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.
        It allows to overwrite the default options.

        .. versionadded:: 1.3

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X : {ndarray, sparse matrix, dataframe}
            The data matrix.
        y : {ndarray, dataframe, series}
            The target.
        frame : dataframe or None
            A dataframe containing both `X` and `y`. `None` if
            `output_array_type != "pandas"`.
        categories : list of str or None
            The names of the features that are categorical. `None` if
            `output_array_type == "pandas"`.
    """
    features_dict = {feature['name']: feature for feature in openml_columns_info}
    if sparse:
        output_type = 'sparse'
    elif as_frame:
        output_type = 'pandas'
    else:
        output_type = 'numpy'
    _verify_target_data_type(features_dict, target_columns)
    for name in target_columns:
        column_info = features_dict[name]
        n_missing_values = int(column_info['number_of_missing_values'])
        if n_missing_values > 0:
            raise ValueError(f"Target column '{column_info['name']}' has {n_missing_values} missing values. Missing values are not supported for target columns.")
    no_retry_exception = None
    if parser == 'pandas':
        from pandas.errors import ParserError
        no_retry_exception = ParserError
    X, y, frame, categories = _retry_with_clean_cache(url, data_home, no_retry_exception)(_load_arff_response)(url, data_home, parser=parser, output_type=output_type, openml_columns_info=features_dict, feature_names_to_select=data_columns, target_names_to_select=target_columns, shape=shape, md5_checksum=md5_checksum, n_retries=n_retries, delay=delay, read_csv_kwargs=read_csv_kwargs)
    return Bunch(data=X, target=y, frame=frame, categories=categories, feature_names=data_columns, target_names=target_columns)