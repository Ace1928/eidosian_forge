import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
def update_metadata(self, metadata: Dict[str, Any]) -> None:
    """Update the metadata stored with this checkpoint.

        This will update any existing metadata stored with this checkpoint.
        """
    existing_metadata = self.get_metadata()
    existing_metadata.update(metadata)
    self.set_metadata(existing_metadata)