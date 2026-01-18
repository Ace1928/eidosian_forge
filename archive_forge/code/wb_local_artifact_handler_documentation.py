import os
from typing import TYPE_CHECKING, Optional, Sequence, Union
import wandb
from wandb import util
from wandb.sdk.artifacts.artifact_instance_cache import artifact_instance_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
Store the file or directory at the given path within the specified artifact.

        Arguments:
            artifact: The artifact doing the storing
            path (str): The path to store
            name (str): If specified, the logical name that should map to `path`

        Returns:
            (list[ArtifactManifestEntry]): A list of manifest entries to store within the artifact
        