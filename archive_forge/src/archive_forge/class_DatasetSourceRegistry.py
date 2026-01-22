import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
class DatasetSourceRegistry:

    def __init__(self):
        self.sources = []

    def register(self, source: DatasetSource):
        """Registers a DatasetSource for use with MLflow Tracking.

        Args:
            source: The DatasetSource to register.
        """
        self.sources.append(source)

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """
        for entrypoint in entrypoints.get_group_all('mlflow.dataset_source'):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(f'Failure attempting to register dataset source with source type "{entrypoint.source_type}": {exc}', stacklevel=2)

    def resolve(self, raw_source: Any, candidate_sources: Optional[List[DatasetSource]]=None) -> DatasetSource:
        """Resolves a raw source object, such as a string URI, to a DatasetSource for use with
        MLflow Tracking.

        Args:
            raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data" or a
                HuggingFace :py:class:`datasets.Dataset` object.
            candidate_sources: A list of DatasetSource classes to consider as potential sources
                when resolving the raw source. Subclasses of the specified candidate sources are
                also considered. If unspecified, all registered sources are considered.

        Raises:
            MlflowException: If no DatasetSource class can resolve the raw source.

        Returns:
            The resolved DatasetSource.
        """
        matching_sources = []
        for source in self.sources:
            if candidate_sources and (not any((issubclass(source, candidate_src) for candidate_src in candidate_sources))):
                continue
            try:
                if source._can_resolve(raw_source):
                    matching_sources.append(source)
            except Exception as e:
                warnings.warn(f"Failed to determine whether {source.__name__} can resolve source information for '{raw_source}'. Exception: {e}", stacklevel=2)
                continue
        if len(matching_sources) > 1:
            source_class_names_str = ', '.join([source.__name__ for source in matching_sources])
            warnings.warn(f'The specified dataset source can be interpreted in multiple ways: {source_class_names_str}. MLflow will assume that this is a {matching_sources[-1].__name__} source.', stacklevel=2)
        for matching_source in reversed(matching_sources):
            try:
                return matching_source._resolve(raw_source)
            except Exception as e:
                warnings.warn(f"Encountered an unexpected error while using {matching_source.__name__} to resolve source information for '{raw_source}'. Exception: {e}", stacklevel=2)
                continue
        raise MlflowException(f'Could not find a source information resolver for the specified dataset source: {raw_source}.', RESOURCE_DOES_NOT_EXIST)

    def get_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
        """Parses and returns a DatasetSource object from its JSON representation.

        Args:
            source_json: The JSON representation of the DatasetSource.
            source_type: The string type of the DatasetSource, which indicates how to parse the
                source JSON.
        """
        for source in reversed(self.sources):
            if source._get_source_type() == source_type:
                return source.from_json(source_json)
        raise MlflowException(f'Could not parse dataset source from JSON due to unrecognized source type: {source_type}.', RESOURCE_DOES_NOT_EXIST)