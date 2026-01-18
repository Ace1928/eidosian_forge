import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, urlparse
from wandb import util
from wandb.errors.term import termlog
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import B64MD5
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
Create an ArtifactManifestEntry from a GCS object.

        Arguments:
            obj: The GCS object
            path: The GCS-style path (e.g.: "gs://bucket/file.txt")
            name: The user assigned name, or None if not specified
            prefix: The prefix to add (will be the same as `path` for directories)
            multi: Whether or not this is a multi-object add.
        