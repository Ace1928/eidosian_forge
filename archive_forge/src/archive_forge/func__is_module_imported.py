import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def _is_module_imported(module_name: str) -> bool:
    return module_name in sys.modules