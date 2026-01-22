import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
def _handle_read_os_error(error: OSError, paths: Union[str, List[str]]) -> str:
    aws_error_pattern = '^(?:(.*)AWS Error \\[code \\d+\\]: No response body\\.(.*))|(?:(.*)AWS Error UNKNOWN \\(HTTP status 400\\) during HeadObject operation: No response body\\.(.*))|(?:(.*)AWS Error ACCESS_DENIED during HeadObject operation: No response body\\.(.*))$'
    if re.match(aws_error_pattern, str(error)):
        if isinstance(paths, str):
            paths = f'"{paths}"'
        raise OSError(f'Failing to read AWS S3 file(s): {paths}. Please check that file exists and has properly configured access. You can also run AWS CLI command to get more detailed error message (e.g., aws s3 ls <file-name>). See https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/index.html and https://docs.ray.io/en/latest/data/creating-datasets.html#reading-from-remote-storage for more information.')
    else:
        raise error