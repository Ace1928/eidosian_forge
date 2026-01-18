from typing import Any, Dict, Mapping, Optional
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import HexMD5, _md5
def to_manifest_json(self) -> Dict:
    """This is the JSON that's stored in wandb_manifest.json.

        If include_local is True we also include the local paths to files. This is
        used to represent an artifact that's waiting to be saved on the current
        system. We don't need to include the local paths in the artifact manifest
        contents.
        """
    contents = {}
    for entry in sorted(self.entries.values(), key=lambda k: k.path):
        json_entry: Dict[str, Any] = {'digest': entry.digest}
        if entry.birth_artifact_id:
            json_entry['birthArtifactID'] = entry.birth_artifact_id
        if entry.ref:
            json_entry['ref'] = entry.ref
        if entry.extra:
            json_entry['extra'] = entry.extra
        if entry.size is not None:
            json_entry['size'] = entry.size
        contents[entry.path] = json_entry
    return {'version': self.__class__.version(), 'storagePolicy': self.storage_policy.name(), 'storagePolicyConfig': self.storage_policy.config() or {}, 'contents': contents}