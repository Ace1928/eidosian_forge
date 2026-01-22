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
class _CustomJsonEncoder(json.JSONEncoder):

    def default(self, o):
        import numpy as np
        import pandas as pd
        if isinstance(o, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)