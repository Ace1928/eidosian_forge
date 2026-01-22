import io
import logging
import time
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
import numpy as np
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.util import _check_import
from ray.data.block import Block, BlockMetadata
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider
from ray.util.annotations import DeveloperAPI
class _ImageFileMetadataProvider(DefaultFileMetadataProvider):

    def _set_encoding_ratio(self, encoding_ratio: int):
        """Set image file encoding ratio, to provide accurate size in bytes metadata."""
        self._encoding_ratio = encoding_ratio

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, rows_per_file: Optional[int], file_sizes: List[Optional[int]]) -> BlockMetadata:
        metadata = super()._get_block_metadata(paths, schema, rows_per_file=rows_per_file, file_sizes=file_sizes)
        if metadata.size_bytes is not None:
            metadata.size_bytes = int(metadata.size_bytes * self._encoding_ratio)
        return metadata