from typing import TYPE_CHECKING, Optional, Sequence, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
class TrackingHandler(StorageHandler):

    def __init__(self, scheme: Optional[str]=None) -> None:
        """Track paths with no modification or special processing.

        Useful when paths being tracked are on file systems mounted at a standardized
        location.

        For example, if the data to track is located on an NFS share mounted on
        `/data`, then it is sufficient to just track the paths.
        """
        self._scheme = scheme or ''

    def can_handle(self, parsed_url: 'ParseResult') -> bool:
        return parsed_url.scheme == self._scheme

    def load_path(self, manifest_entry: ArtifactManifestEntry, local: bool=False) -> Union[URIStr, FilePathStr]:
        if local:
            url = urlparse(manifest_entry.ref)
            raise ValueError(f'Cannot download file at path {str(manifest_entry.ref)}, scheme {str(url.scheme)} not recognized')
        return FilePathStr(manifest_entry.path)

    def store_path(self, artifact: 'Artifact', path: Union[URIStr, FilePathStr], name: Optional[StrPath]=None, checksum: bool=True, max_objects: Optional[int]=None) -> Sequence[ArtifactManifestEntry]:
        url = urlparse(path)
        if name is None:
            raise ValueError('You must pass name="<entry_name>" when tracking references with unknown schemes. ref: %s' % path)
        termwarn('Artifact references with unsupported schemes cannot be checksummed: %s' % path)
        name = name or url.path[1:]
        return [ArtifactManifestEntry(path=name, ref=path, digest=path)]