import logging
import os
import pathlib
import sys
import traceback
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.utils import reraise
from mlflow.utils.annotations import deprecated, keyword_only
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
@deprecated(alternative='mlflow.onnx', since='2.6.0')
@keyword_only
def add_to_model(mlflow_model, path, spark_model, sample_input):
    """
    Add the MLeap flavor to an existing MLflow model.

    Args:
        mlflow_model: :py:mod:`mlflow.models.Model` to which this flavor is being added.
        path: Path of the model to which this flavor is being added.
        spark_model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
            cannot contain any custom transformers.
        sample_input: Sample PySpark DataFrame input that the model can evaluate. This is
            required by MLeap for data schema inference.
    """
    import mleap.version
    from mleap.pyspark.spark_support import SimpleSparkSerializer
    from py4j.protocol import Py4JError
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql import DataFrame
    if not isinstance(spark_model, PipelineModel):
        raise Exception('Not a PipelineModel. MLeap can save only PipelineModels.')
    if sample_input is None:
        raise Exception('A sample input must be specified in order to add the MLeap flavor.')
    if not isinstance(sample_input, DataFrame):
        raise Exception(f'The sample input must be a PySpark dataframe of type `{DataFrame.__module__}`')
    path = os.path.abspath(path)
    mleap_path_full = os.path.join(path, 'mleap')
    mleap_datapath_sub = os.path.join('mleap', 'model')
    mleap_datapath_full = os.path.join(path, mleap_datapath_sub)
    if os.path.exists(mleap_path_full):
        raise Exception(f'MLeap model data path already exists at: {mleap_path_full}')
    os.makedirs(mleap_path_full)
    dataset = spark_model.transform(sample_input)
    if is_windows():
        model_path = 'file://' + str(pathlib.Path(mleap_datapath_full).as_posix())
    else:
        model_path = path_to_local_file_uri(mleap_datapath_full)
    try:
        spark_model.serializeToBundle(path=model_path, dataset=dataset)
    except Py4JError:
        _handle_py4j_error(MLeapSerializationException, 'MLeap encountered an error while serializing the model. Ensure that the model is compatible with MLeap (i.e does not contain any custom transformers).')
    try:
        mleap_version = mleap.version.__version__
        _logger.warning('Detected old mleap version %s. Support for logging models in mleap format with mleap versions 0.15.0 and below is deprecated and will be removed in a future MLflow release. Please upgrade to a newer mleap version.', mleap_version)
    except AttributeError:
        mleap_version = mleap.version
    mlflow_model.add_flavor(FLAVOR_NAME, mleap_version=mleap_version, model_data=mleap_datapath_sub)