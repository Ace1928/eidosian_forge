import json
import logging
import os
import posixpath
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import mleap, pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.spark import (
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
from mlflow.utils import databricks_utils
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
def _get_or_create_sparksession(model_path=None):
    """Check if SparkSession running and get it.

    If none exists, create a new one using jars in model_path. If model_path not defined, rely on
    nlp.start() to create a new one using johnsnowlabs Jar resolution method. See
    https://nlp.johnsnowlabs.com/docs/en/jsl/start-a-sparksession and
    https://nlp.johnsnowlabs.com/docs/en/jsl/install_advanced.

    Args:
        model_path:

    Returns:

    """
    from johnsnowlabs import nlp
    from mlflow.utils._spark_utils import _get_active_spark_session
    _validate_env_vars()
    spark = _get_active_spark_session()
    if spark is None:
        spark_conf = {}
        spark_conf['spark.python.worker.reuse'] = 'true'
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        if model_path:
            jar_paths, license_path = _fetch_deps_from_path(model_path)
            if license_path:
                with open(license_path) as f:
                    loaded_license = json.load(f)
                    os.environ.update({k: str(v) for k, v in loaded_license.items() if v is not None})
                    os.environ['JSL_NLP_LICENSE'] = loaded_license['HC_LICENSE']
            _logger.info('Starting a new Session with Jars: %s', jar_paths)
            spark = nlp.start(nlp=False, spark_nlp=False, jar_paths=jar_paths, json_license_path=license_path, create_jsl_home_if_missing=False, spark_conf=spark_conf)
        else:
            spark = nlp.start()
    return spark