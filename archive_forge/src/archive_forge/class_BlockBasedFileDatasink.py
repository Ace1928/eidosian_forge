import posixpath
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import _is_local_scheme, call_with_retry
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource.block_path_provider import BlockWritePathProvider
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.filename_provider import (
from ray.data.datasource.path_util import _resolve_paths_and_filesystem
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class BlockBasedFileDatasink(_FileDatasink):
    """A datasink that writes multiple rows to each file.

    Subclasses must implement ``write_block_to_file`` and call the superclass
    constructor.

    Examples:
        .. testcode::

            class CSVDatasink(BlockBasedFileDatasink):
                def __init__(self, path: str):
                    super().__init__(path, file_format="csv")

                def write_block_to_file(self, block: BlockAccessor, file: "pyarrow.NativeFile"):
                    from pyarrow import csv
                    csv.write_csv(block.to_arrow(), file)
    """

    def write_block_to_file(self, block: BlockAccessor, file: 'pyarrow.NativeFile'):
        """Write a block of data to a file.

        Args:
            block: The block to write.
            file: The file to write the block to.
        """
        raise NotImplementedError

    def write_block(self, block: BlockAccessor, block_index: int, ctx: TaskContext):
        if self.filename_provider is not None:
            filename = self.filename_provider.get_filename_for_block(block, ctx.task_idx, block_index)
            write_path = posixpath.join(self.path, filename)
        else:
            write_path = self.block_path_provider(self.path, filesystem=self.filesystem, dataset_uuid=self.dataset_uuid, task_index=ctx.task_idx, block_index=block_index, file_format=self.file_format)

        def write_block_to_path():
            with self.open_output_stream(write_path) as file:
                self.write_block_to_file(block, file)
        logger.get_logger().debug(f'Writing {write_path} file.')
        call_with_retry(write_block_to_path, match=DataContext.get_current().write_file_retry_on_errors, description=f"write '{write_path}'", max_attempts=WRITE_FILE_MAX_ATTEMPTS, max_backoff_s=WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS)