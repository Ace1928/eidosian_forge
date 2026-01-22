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
def _parse_uri(self, uri: str) -> Tuple[str, str, Optional[str]]:
    url = urlparse(uri)
    bucket = url.netloc
    key = url.path[1:]
    version = url.fragment if url.fragment else None
    return (bucket, key, version)