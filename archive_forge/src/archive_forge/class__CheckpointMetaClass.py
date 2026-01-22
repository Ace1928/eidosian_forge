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
class _CheckpointMetaClass(type):

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError as exc:
            if item in {'from_dict', 'to_dict', 'from_bytes', 'to_bytes', 'get_internal_representation'}:
                raise _get_migration_error(item) from exc
            elif item in {'from_uri', 'to_uri', 'uri'}:
                raise _get_uri_error(item) from exc
            elif item in {'get_preprocessor', 'set_preprocessor'}:
                raise _get_preprocessor_error(item) from exc
            raise exc