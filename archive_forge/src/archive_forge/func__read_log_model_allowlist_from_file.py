import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _read_log_model_allowlist_from_file(allowlist_file):

    def _parse_allowlist_file(line_iter):
        allowlist = set()
        for line in line_iter:
            stripped = line.strip()
            is_blankline_or_comment = stripped == '' or stripped.startswith('#')
            if not is_blankline_or_comment:
                allowlist.add(stripped)
        return allowlist
    url_parsed = urlparse(allowlist_file)
    scheme = url_parsed.scheme
    path = url_parsed.path
    if is_windows() and (not url_parsed.hostname):
        path = scheme + '://' + path
        scheme = ''
    if scheme in ('file', ''):
        if not os.path.exists(path):
            raise MlflowException.invalid_parameter_value(f'{allowlist_file} does not exist')
        with open(allowlist_file) as f:
            return _parse_allowlist_file(f)
    else:
        host_creds = MlflowHostCreds(host=scheme + '://' + (url_parsed.hostname or ''), username=url_parsed.username, password=url_parsed.password)
        response = http_request(host_creds=host_creds, endpoint=path, method='GET')
        augmented_raise_for_status(response)
        return _parse_allowlist_file(response.iter_lines(decode_unicode=True))