import json
import pathlib
import pickle
from collections import namedtuple
from json import JSONDecodeError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationArtifact
from mlflow.utils.annotations import developer_stable
from mlflow.utils.proto_json_utils import NumpyEncoder
@developer_stable
class TextEvaluationArtifact(EvaluationArtifact):

    def _save(self, output_artifact_path):
        with open(output_artifact_path, 'w') as f:
            f.write(self._content)

    def _load_content_from_file(self, local_artifact_path):
        with open(local_artifact_path) as f:
            self._content = f.read()
        return self._content