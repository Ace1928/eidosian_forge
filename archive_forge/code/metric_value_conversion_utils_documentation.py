import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
Wrapper function that checks if specified `module_name` is already imported before
    invoking wrapped function.
    