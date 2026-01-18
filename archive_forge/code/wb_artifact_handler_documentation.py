import os
from typing import TYPE_CHECKING, Optional, Sequence, Union
from urllib.parse import urlparse
import wandb
from wandb import util
from wandb.apis import PublicApi
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, hex_to_b64_id
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
Store the file or directory at the given path into the specified artifact.

        Recursively resolves the reference until the result is a concrete asset.

        Arguments:
            artifact: The artifact doing the storing path (str): The path to store name
            (str): If specified, the logical name that should map to `path`

        Returns:
            (list[ArtifactManifestEntry]): A list of manifest entries to store within
            the artifact
        