import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _valid_sample(sample: Dict[str, Any]):
    """Check whether a sample is valid.

    Args:
        sample: sample to be checked
    """
    return sample is not None and isinstance(sample, dict) and (len(list(sample.keys())) > 0) and (not sample.get('__bad__', False))