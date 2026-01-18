from typing import TYPE_CHECKING, Optional, Sequence, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
Track paths with no modification or special processing.

        Useful when paths being tracked are on file systems mounted at a standardized
        location.

        For example, if the data to track is located on an NFS share mounted on
        `/data`, then it is sufficient to just track the paths.
        