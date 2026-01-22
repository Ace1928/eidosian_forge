import sys
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
def _try_get_item(x):
    try:
        return x.item()
    except Exception as e:
        raise MlflowException(f'Failed to convert metric value to float: {e}', error_code=INVALID_PARAMETER_VALUE)