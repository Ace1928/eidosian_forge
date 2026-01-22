from pathlib import PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, parse_qsl, urlparse
import wandb
from wandb import util
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import ETag
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def _get_credential(self, account_url: str) -> Union['azure.identity.DefaultAzureCredential', str]:
    if wandb.run and wandb.run.settings.azure_account_url_to_access_key is not None and (account_url in wandb.run.settings.azure_account_url_to_access_key):
        return wandb.run.settings.azure_account_url_to_access_key[account_url]
    return self._get_module('azure.identity').DefaultAzureCredential()