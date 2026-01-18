import json
import logging
import os
from datetime import datetime
from io import StringIO
from typing import ForwardRef, get_args, get_origin
from mlflow.exceptions import MlflowException
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data

    Generate predictions in json format using a saved MLflow model. For information about the input
    data formats accepted by this function, see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    Args:
        model_uri: URI to the model. A local path, a local or remote URI e.g. runs:/, s3://.
        input_data: Input data for prediction. Must be valid input for the PyFunc model. Refer
            to the :py:func:`mlflow.pyfunc.PyFuncModel.predict()` for the supported input types.
        input_path: Path to a file containing input data. If provided, 'input_data' must be None.
        content_type: Content type of the input data. Can be one of {‘json’, ‘csv’}.
        output_path: File to output results to as json. If not provided, output to stdout.
        env_manager: Specify a way to create an environment for MLmodel inference:

            - "virtualenv" (default): use virtualenv (and pyenv for Python version management)
            - "local": use the local environment
            - "conda": use conda

        install_mlflow: If specified and there is a conda or virtualenv environment to be activated
            mlflow will be installed into the environment after it has been activated. The version
            of installed mlflow will be the same as the one used to invoke this command.
        pip_requirements_override: If specified, install the specified python dependencies to the
            model inference environment. This is particularly useful when you want to add extra
            dependencies or try different versions of the dependencies defined in the logged model.

    Code example:

    .. code-block:: python

        import mlflow

        run_id = "..."

        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data={"x": 1, "y": 2},
            content_type="json",
        )

        # Run prediction with additional pip dependencies
        mlflow.models.predict(
            model_uri=f"runs:/{run_id}/model",
            input_data='{"x": 1, "y": 2}',
            content_type="json",
            pip_requirements_override=["scikit-learn==0.23.2"],
        )

    