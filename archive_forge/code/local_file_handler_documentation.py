import os
import shutil
import time
from typing import TYPE_CHECKING, Optional, Sequence, Union
from urllib.parse import ParseResult
from wandb import util
from wandb.errors.term import termlog
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64, md5_string
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
Track files or directories on a local filesystem.

        Expand directories to create an entry for each file contained.
        