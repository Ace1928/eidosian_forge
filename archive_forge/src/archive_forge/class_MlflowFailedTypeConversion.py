import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
class MlflowFailedTypeConversion(MlflowInvalidInputException):

    def __init__(self, col_name, col_type, ex):
        super().__init__(message=f"Data is not compatible with model signature. Failed to convert column {col_name} to type '{col_type}'. Error: '{ex!r}'")