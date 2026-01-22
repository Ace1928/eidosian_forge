import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
import mlflow.tracking
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _OnnxModelWrapper:

    def __init__(self, path, providers=None):
        import onnxruntime
        local_path = str(Path(path).parent)
        model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
        if 'providers' in model_meta.flavors.get(FLAVOR_NAME).keys():
            providers = model_meta.flavors.get(FLAVOR_NAME)['providers']
        else:
            providers = ONNX_EXECUTION_PROVIDERS
        sess_options = onnxruntime.SessionOptions()
        options = model_meta.flavors.get(FLAVOR_NAME).get('onnx_session_options')
        if options:
            if (inter_op_num_threads := options.get('inter_op_num_threads')):
                sess_options.inter_op_num_threads = inter_op_num_threads
            if (intra_op_num_threads := options.get('intra_op_num_threads')):
                sess_options.intra_op_num_threads = intra_op_num_threads
            if (execution_mode := options.get('execution_mode')):
                if execution_mode.upper() == 'SEQUENTIAL':
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                elif execution_mode.upper() == 'PARALLEL':
                    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            if (graph_optimization_level := options.get('graph_optimization_level')):
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(graph_optimization_level)
            if (extra_session_config := options.get('extra_session_config')):
                for key, value in extra_session_config.items():
                    sess_options.add_session_config_entry(key, value)
        try:
            self.rt = onnxruntime.InferenceSession(path, sess_options=sess_options)
        except ValueError:
            self.rt = onnxruntime.InferenceSession(path, providers=providers, sess_options=sess_options)
        assert len(self.rt.get_inputs()) >= 1
        self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
        self.output_names = [outp.name for outp in self.rt.get_outputs()]

    def _cast_float64_to_float32(self, feeds):
        for input_name, input_type in self.inputs:
            if input_type == 'tensor(float)':
                feed = feeds.get(input_name)
                if feed is not None and feed.dtype == np.float64:
                    feeds[input_name] = feed.astype(np.float32)
        return feeds

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Either a pandas DataFrame, numpy.ndarray or a dictionary.
                Dictionary input is expected to be a valid ONNX model feed dictionary.

                Numpy array input is supported iff the model has a single tensor input and is
                converted into an ONNX feed dictionary with the appropriate key.

                Pandas DataFrame is converted to ONNX inputs as follows:
                    - If the underlying ONNX model only defines a *single* input tensor, the
                      DataFrame's values are converted to a NumPy array representation using the
                      `DataFrame.values()
                      <https://pandas.pydata.org/pandas-docs/stable/reference/api/
                      pandas.DataFrame.values.html#pandas.DataFrame.values>`_ method.
                    - If the underlying ONNX model defines *multiple* input tensors, each column
                      of the DataFrame is converted to a NumPy array representation.

                For more information about the ONNX Runtime, see
                `<https://github.com/microsoft/onnxruntime>`_.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                                        release without warning.

        Returns:
            Model predictions. If the input is a pandas.DataFrame, the predictions are returned
            in a pandas.DataFrame. If the input is a numpy array or a dictionary the
            predictions are returned in a dictionary.
        """
        if isinstance(data, dict):
            feed_dict = data
        elif isinstance(data, np.ndarray):
            if len(self.inputs) != 1:
                inputs = [x[0] for x in self.inputs]
                raise MlflowException(f'Unable to map numpy array input to the expected model input. Numpy arrays can only be used as input for MLflow ONNX models that have a single input. This model requires {len(self.inputs)} inputs. Please pass in data as either a dictionary or a DataFrame with the following tensors: {inputs}.')
            feed_dict = {self.inputs[0][0]: data}
        elif isinstance(data, pd.DataFrame):
            if len(self.inputs) > 1:
                feed_dict = {name: data[name].values for name, _ in self.inputs}
            else:
                feed_dict = {self.inputs[0][0]: data.values}
        else:
            raise TypeError(f"Input should be a dictionary or a numpy array or a pandas.DataFrame, got '{type(data)}'")
        feed_dict = self._cast_float64_to_float32(feed_dict)
        predicted = self.rt.run(self.output_names, feed_dict)
        if isinstance(data, pd.DataFrame):

            def format_output(data):
                data = np.asarray(data)
                return data.reshape(-1)
            return pd.DataFrame.from_dict({c: format_output(p) for c, p in zip(self.output_names, predicted)})
        else:
            return dict(zip(self.output_names, predicted))