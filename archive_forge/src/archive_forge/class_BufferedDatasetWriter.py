import logging
from rasterio._base import get_dataset_driver, driver_can_create, driver_can_create_copy
from rasterio._io import (
from rasterio.windows import WindowMethodsMixin
from rasterio.env import Env, ensure_env
from rasterio.transform import TransformMethodsMixin
from rasterio._path import _UnparsedPath
class BufferedDatasetWriter(BufferedDatasetWriterBase, WindowMethodsMixin, TransformMethodsMixin):
    """Maintains data and metadata in a buffer, writing to disk or
    network only when `close()` is called.

    This allows incremental updates to datasets using formats that don't
    otherwise support updates, such as JPEG.
    """

    def __repr__(self):
        return "<{} BufferedDatasetWriter name='{}' mode='{}'>".format(self.closed and 'closed' or 'open', self.name, self.mode)