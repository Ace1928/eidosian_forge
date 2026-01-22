import getpass
import sys
from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.credentials import read_mlflow_creds
from mlflow.utils.mlflow_tags import (
def _get_source_name():
    main_file = _get_main_file()
    if main_file is not None:
        return main_file
    return '<console>'