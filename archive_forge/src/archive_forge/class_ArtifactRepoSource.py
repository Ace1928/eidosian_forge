import re
import warnings
from pathlib import Path
from typing import Any, Dict, TypeVar
from urllib.parse import urlparse
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_registered_artifact_repositories
from mlflow.utils.uri import is_local_uri
class ArtifactRepoSource(FileSystemDatasetSource):

    def __init__(self, uri: str):
        self._uri = uri

    @property
    def uri(self):
        """
            The URI with scheme '{scheme}' referring to the dataset source filesystem location.

            Returns
                The URI with scheme '{scheme}' referring to the dataset source filesystem
                location.
            """
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return source_type

    def load(self, dst_path=None) -> str:
        """
            Downloads the dataset source to the local filesystem.

            Args:
                dst_path: Path of the local filesystem destination directory to which to download
                    the dataset source. If the directory does not exist, it is created. If
                    unspecified, the dataset source is downloaded to a new uniquely-named
                    directory on the local filesystem, unless the dataset source already
                    exists on the local filesystem, in which case its local path is
                    returned directly.

            Returns:
                The path to the downloaded dataset source on the local filesystem.
            """
        return download_artifacts(artifact_uri=self.uri, dst_path=dst_path)

    @staticmethod
    def _can_resolve(raw_source: Any):
        is_local_source_type = ArtifactRepoSource._get_source_type() == 'local'
        if not isinstance(raw_source, str) and (not isinstance(raw_source, Path) and is_local_source_type):
            return False
        try:
            if is_local_source_type:
                return is_local_uri(str(raw_source), is_tracking_or_registry_uri=False)
            else:
                parsed_source = urlparse(str(raw_source))
                return parsed_source.scheme == scheme
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> DatasetForArtifactRepoSourceType:
        return cls(str(raw_source))

    def to_dict(self) -> Dict[Any, Any]:
        """
            Returns:
                A JSON-compatible dictionary representation of the {dataset_source_name}.
            """
        return {'uri': self.uri}

    @classmethod
    def from_dict(cls, source_dict: Dict[Any, Any]) -> DatasetForArtifactRepoSourceType:
        uri = source_dict.get('uri')
        if uri is None:
            raise MlflowException(f'Failed to parse {dataset_source_name}. Missing expected key: "uri"', INVALID_PARAMETER_VALUE)
        return cls(uri=uri)