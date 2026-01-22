import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import DataType
from mlflow.types.schema import (
class TensorsNotSupportedException(MlflowException):

    def __init__(self, msg):
        super().__init__(f'Multidimensional arrays (aka tensors) are not supported. {msg}')