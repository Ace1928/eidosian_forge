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
class RowBasedFileDatasink(_FileDatasink):
    """A datasink that writes one row to each file.

    Subclasses must implement ``write_row_to_file`` and call the superclass constructor.

    Examples:
        .. testcode::

            import io
            import numpy as np
            from PIL import Image
            from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
            from ray.data.datasource import FileBasedDatasource

            class ImageDatasource(FileBasedDatasource):
                def __init__(self, paths):
                    super().__init__(
                        paths,
                        file_extensions=["png", "jpg", "jpeg", "bmp", "gif", "tiff"],
                    )

                def _read_stream(self, f, path):
                    data = f.readall()
                    image = Image.open(io.BytesIO(data))

                    builder = DelegatingBlockBuilder()
                    array = np.array(image)
                    item = {"image": array}
                    builder.add(item)
                    yield builder.build()
    """

    def write_row_to_file(self, row: Dict[str, Any], file: 'pyarrow.NativeFile'):
        """Write a row to a file.

        Args:
            row: The row to write.
            file: The file to write the row to.
        """
        raise NotImplementedError

    def write_block(self, block: BlockAccessor, block_index: int, ctx: TaskContext):
        for row_index, row in enumerate(block.iter_rows(public_row_format=False)):
            if self.filename_provider is not None:
                filename = self.filename_provider.get_filename_for_row(row, ctx.task_idx, block_index, row_index)
            else:
                filename = f'{self.dataset_uuid}_{ctx.task_idx:06}_{block_index:06}_{row_index:06}.{self.file_format}'
            write_path = posixpath.join(self.path, filename)

            def write_row_to_path():
                with self.open_output_stream(write_path) as file:
                    self.write_row_to_file(row, file)
            logger.get_logger().debug(f'Writing {write_path} file.')
            call_with_retry(write_row_to_path, match=DataContext.get_current().write_file_retry_on_errors, description=f"write '{write_path}'", max_attempts=WRITE_FILE_MAX_ATTEMPTS, max_backoff_s=WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS)