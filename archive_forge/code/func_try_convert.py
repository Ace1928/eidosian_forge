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
def try_convert(self, o):
    import numpy as np
    import pandas as pd

    def encode_binary(x):
        return base64.encodebytes(x).decode('ascii')
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            return ([self.try_convert(x)[0] for x in o.tolist()], True)
        elif o.dtype == np.bytes_:
            return (np.vectorize(encode_binary)(o), True)
        else:
            return (o.tolist(), True)
    if isinstance(o, np.generic):
        return (o.item(), True)
    if isinstance(o, (bytes, bytearray)):
        return (encode_binary(o), True)
    if isinstance(o, np.datetime64):
        return (np.datetime_as_string(o), True)
    if isinstance(o, (pd.Timestamp, datetime.date, datetime.datetime, datetime.time)):
        return (o.isoformat(), True)
    return (o, False)