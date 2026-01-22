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
def _save_jars_and_lic(dst_dir, store_license=False):
    from johnsnowlabs.auto_install.jsl_home import get_install_suite_from_jsl_home
    from johnsnowlabs.py_models.jsl_secrets import JslSecrets
    deps_data_path = Path(dst_dir) / _JOHNSNOWLABS_MODEL_PATH_SUB / 'jars.jsl'
    deps_data_path.mkdir(parents=True, exist_ok=True)
    suite = get_install_suite_from_jsl_home(False, visual=_JOHNSNOWLABS_ENV_VISUAL_SECRET in os.environ)
    if suite.hc.get_java_path():
        shutil.copy2(suite.hc.get_java_path(), deps_data_path / 'hc_jar.jar')
    if suite.nlp.get_java_path():
        shutil.copy2(suite.nlp.get_java_path(), deps_data_path / 'os_jar.jar')
    if suite.ocr.get_java_path():
        shutil.copy2(suite.ocr.get_java_path(), deps_data_path / 'visual_nlp.jar')
    if store_license:
        secrets = JslSecrets.build_or_try_find_secrets()
        if secrets.HC_LICENSE:
            deps_data_path.joinpath('license.json').write(secrets.json())