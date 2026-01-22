import logging
import os
from typing import Any, Dict, Optional
import pandas as pd
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _SpacyModelWrapper:

    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]]=None):
        """Only works for predicting using text categorizer.
        Not suitable for other pipeline components (e.g: parser)

        Args:
            dataframe: pandas dataframe containing texts to be categorized
                       expected shape is (n_rows,1 column)
            params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        Returns:
            dataframe with predictions
        """
        if len(dataframe.columns) != 1:
            raise MlflowException('Shape of input dataframe must be (n_rows, 1column)')
        return pd.DataFrame({'predictions': dataframe.iloc[:, 0].apply(lambda text: self.spacy_model(text).cats)})