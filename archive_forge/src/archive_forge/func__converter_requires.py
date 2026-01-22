import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def _converter_requires(module_name: str):
    """Wrapper function that checks if specified `module_name` is already imported before
    invoking wrapped function.
    """

    def decorator(func):

        def wrapper(x):
            if not _is_module_imported(module_name):
                return x
            return func(x)
        return wrapper
    return decorator