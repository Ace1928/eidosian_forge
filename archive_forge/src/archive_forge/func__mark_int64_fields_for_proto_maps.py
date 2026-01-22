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
def _mark_int64_fields_for_proto_maps(proto_map, value_field_type):
    """Converts a proto map to JSON, preserving only int64-related fields."""
    json_dict = {}
    for key, value in proto_map.items():
        if value_field_type == FieldDescriptor.TYPE_MESSAGE:
            json_dict[key] = _mark_int64_fields(value)
        elif value_field_type in _PROTOBUF_INT64_FIELDS:
            json_dict[key] = int(value)
        elif isinstance(key, int):
            json_dict[key] = value
    return json_dict