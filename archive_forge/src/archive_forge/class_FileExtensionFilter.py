import io
import pathlib
import posixpath
import warnings
from typing import (
import numpy as np
import ray
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasource import Datasource, ReadTask, WriteResult
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.filename_provider import (
from ray.data.datasource.partitioning import (
from ray.data.datasource.path_util import (
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@Deprecated
@PublicAPI(stability='beta')
class FileExtensionFilter(PathPartitionFilter):
    """A file-extension-based path filter that filters files that don't end
    with the provided extension(s).

    Attributes:
        file_extensions: File extension(s) of files to be included in reading.
        allow_if_no_extension: If this is True, files without any extensions
            will be included in reading.

    """

    def __init__(self, file_extensions: Union[str, List[str]], allow_if_no_extension: bool=False):
        warnings.warn('`FileExtensionFilter` is deprecated. Instead, set the `file_extensions` parameter of `read_xxx()` APIs.', DeprecationWarning)
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        self.extensions = [f'.{ext.lower()}' for ext in file_extensions]
        self.allow_if_no_extension = allow_if_no_extension

    def _file_has_extension(self, path: str):
        suffixes = [suffix.lower() for suffix in pathlib.Path(path).suffixes]
        if not suffixes:
            return self.allow_if_no_extension
        return any((ext in suffixes for ext in self.extensions))

    def __call__(self, paths: List[str]) -> List[str]:
        return [path for path in paths if self._file_has_extension(path)]

    def __str__(self):
        return f'{type(self).__name__}(extensions={self.extensions}, allow_if_no_extensions={self.allow_if_no_extension})'

    def __repr__(self):
        return str(self)