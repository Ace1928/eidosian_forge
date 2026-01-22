import json
import keyword
import logging
import math
import operator
import os
import pathlib
import signal
import struct
import sys
import urllib
import urllib.parse
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from decimal import Decimal
from types import FunctionType
from typing import Any, Dict, Optional
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.validation import (
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _get_fully_qualified_class_name, insecure_hash
from mlflow.utils.annotations import developer_stable, experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.string_utils import generate_feature_name_if_not_string
@developer_stable
class EvaluationArtifact(metaclass=ABCMeta):
    """
    A model evaluation artifact containing an artifact uri and content.
    """

    def __init__(self, uri, content=None):
        self._uri = uri
        self._content = content

    @abstractmethod
    def _load_content_from_file(self, local_artifact_path):
        """
        Abstract interface to load the content from local artifact file path,
        and return the loaded content.
        """
        pass

    def _load(self, local_artifact_path=None):
        """
        If ``local_artifact_path`` is ``None``, download artifact from the artifact uri.
        Otherwise, load artifact content from the specified path. Assign the loaded content to
        ``self._content``, and return the loaded content.
        """
        if local_artifact_path is not None:
            self._content = self._load_content_from_file(local_artifact_path)
        else:
            with TempDir() as temp_dir:
                temp_dir_path = temp_dir.path()
                _download_artifact_from_uri(self._uri, temp_dir_path)
                local_artifact_file = temp_dir.path(os.listdir(temp_dir_path)[0])
                self._content = self._load_content_from_file(local_artifact_file)
        return self._content

    @abstractmethod
    def _save(self, output_artifact_path):
        """Save artifact content into specified path."""
        pass

    @property
    def content(self):
        """
        The content of the artifact (representation varies)
        """
        if self._content is None:
            self._load()
        return self._content

    @property
    def uri(self) -> str:
        """
        The URI of the artifact
        """
        return self._uri

    def __repr__(self):
        return f"{self.__class__.__name__}(uri='{self.uri}')"