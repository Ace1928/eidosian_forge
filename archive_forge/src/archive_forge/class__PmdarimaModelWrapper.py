import logging
import os
import pickle
import warnings
from typing import Any, Dict, Optional
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _PmdarimaModelWrapper:

    def __init__(self, pmdarima_model):
        import pmdarima
        self.pmdarima_model = pmdarima_model
        self._pmdarima_version = pmdarima.__version__

    def predict(self, dataframe, params: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        df_schema = dataframe.columns.values.tolist()
        if len(dataframe) > 1:
            raise MlflowException(f'The provided prediction pd.DataFrame contains {len(dataframe)} rows. Only 1 row should be supplied.', error_code=INVALID_PARAMETER_VALUE)
        attrs = dataframe.to_dict(orient='index').get(0)
        n_periods = attrs.get('n_periods', None)
        if not n_periods:
            raise MlflowException(f'The provided prediction configuration pd.DataFrame columns ({df_schema}) do not contain the required column `n_periods` for specifying future prediction periods to generate.', error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(n_periods, int):
            raise MlflowException(f'The provided `n_periods` value {n_periods} must be an integer.provided type: {type(n_periods)}', error_code=INVALID_PARAMETER_VALUE)
        exogenous_regressor = attrs.get('X', None)
        if exogenous_regressor and Version(self._pmdarima_version) < Version('1.8.0'):
            warnings.warn(f"An exogenous regressor element was provided in column 'X'. This is supported only in pmdarima version >= 1.8.0. Installed version: {self._pmdarima_version}")
        return_conf_int = attrs.get('return_conf_int', False)
        alpha = attrs.get('alpha', 0.05)
        if not isinstance(n_periods, int):
            raise MlflowException('The prediction DataFrame must contain a column `n_periods` with an integer value for number of future periods to predict.', error_code=INVALID_PARAMETER_VALUE)
        if Version(self._pmdarima_version) >= Version('1.8.0'):
            raw_predictions = self.pmdarima_model.predict(n_periods=n_periods, X=exogenous_regressor, return_conf_int=return_conf_int, alpha=alpha)
        else:
            raw_predictions = self.pmdarima_model.predict(n_periods=n_periods, return_conf_int=return_conf_int, alpha=alpha)
        if return_conf_int:
            ci_low, ci_high = list(zip(*raw_predictions[1]))
            predictions = pd.DataFrame.from_dict({'yhat': raw_predictions[0], 'yhat_lower': ci_low, 'yhat_upper': ci_high})
        else:
            predictions = pd.DataFrame.from_dict({'yhat': raw_predictions})
        return predictions