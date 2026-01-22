import getpass
import sys
from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.credentials import read_mlflow_creds
from mlflow.utils.mlflow_tags import (
Get the current computer username.