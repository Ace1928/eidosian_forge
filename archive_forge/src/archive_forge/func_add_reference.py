import atexit
import concurrent.futures
import contextlib
import json
import multiprocessing.dummy
import os
import re
import shutil
import tempfile
import time
from copy import copy
from datetime import datetime, timedelta
from functools import partial
from pathlib import PurePosixPath
from typing import (
from urllib.parse import urlparse
import requests
import wandb
from wandb import data_types, env, util
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public import ArtifactCollection, ArtifactFiles, RetryingClient, Run
from wandb.data_types import WBValue
from wandb.errors.term import termerror, termlog, termwarn
from wandb.sdk.artifacts.artifact_download_logger import ArtifactDownloadLogger
from wandb.sdk.artifacts.artifact_instance_cache import artifact_instance_cache
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.artifact_manifests.artifact_manifest_v1 import (
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.artifacts.artifact_ttl import ArtifactTTL
from wandb.sdk.artifacts.exceptions import (
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.artifacts.storage_layout import StorageLayout
from wandb.sdk.artifacts.storage_policies import WANDB_STORAGE_POLICY
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.data_types._dtypes import Type as WBType
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, retry, runid, telemetry
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, md5_file_b64
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
from wandb.sdk.lib.runid import generate_id
from wandb.util import get_core_path
from wandb_gql import gql  # noqa: E402
def add_reference(self, uri: Union[ArtifactManifestEntry, str], name: Optional[StrPath]=None, checksum: bool=True, max_objects: Optional[int]=None) -> Sequence[ArtifactManifestEntry]:
    """Add a reference denoted by a URI to the artifact.

        Unlike files or directories that you add to an artifact, references are not
        uploaded to W&B. For more information,
        see [Track external files](https://docs.wandb.ai/guides/artifacts/track-external-files).

        By default, the following schemes are supported:

        - http(s): The size and digest of the file will be inferred by the
          `Content-Length` and the `ETag` response headers returned by the server.
        - s3: The checksum and size are pulled from the object metadata. If bucket
          versioning is enabled, then the version ID is also tracked.
        - gs: The checksum and size are pulled from the object metadata. If bucket
          versioning is enabled, then the version ID is also tracked.
        - https, domain matching `*.blob.core.windows.net` (Azure): The checksum and size
          are be pulled from the blob metadata. If storage account versioning is
          enabled, then the version ID is also tracked.
        - file: The checksum and size are pulled from the file system. This scheme
          is useful if you have an NFS share or other externally mounted volume
          containing files you wish to track but not necessarily upload.

        For any other scheme, the digest is just a hash of the URI and the size is left
        blank.

        Arguments:
            uri: The URI path of the reference to add. The URI path can be an object
                returned from `Artifact.get_entry` to store a reference to another
                artifact's entry.
            name: The path within the artifact to place the contents of this reference.
            checksum: Whether or not to checksum the resource(s) located at the
                reference URI. Checksumming is strongly recommended as it enables
                automatic integrity validation, however it can be disabled to speed up
                artifact creation.
            max_objects: The maximum number of objects to consider when adding a
                reference that points to directory or bucket store prefix. By default,
                the maximum number of objects allowed for Amazon S3 and
                GCS is 10,000. Other URI schemas do not have a maximum.

        Returns:
            The added manifest entries.

        Raises:
            ArtifactFinalizedError: You cannot make changes to the current artifact
            version because it is finalized. Log a new artifact version instead.
        """
    self._ensure_can_add()
    if name is not None:
        name = LogicalPath(name)
    if isinstance(uri, ArtifactManifestEntry):
        uri_str = uri.ref_url()
    elif isinstance(uri, str):
        uri_str = uri
    url = urlparse(str(uri_str))
    if not url.scheme:
        raise ValueError('References must be URIs. To reference a local file, use file://')
    manifest_entries = self._storage_policy.store_reference(self, URIStr(uri_str), name=name, checksum=checksum, max_objects=max_objects)
    for entry in manifest_entries:
        self.manifest.add_entry(entry)
    return manifest_entries