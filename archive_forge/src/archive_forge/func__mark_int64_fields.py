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
def _mark_int64_fields(proto_message):
    """Converts a proto message to JSON, preserving only int64-related fields."""
    json_dict = {}
    for field, value in proto_message.ListFields():
        if field.type == FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry:
            json_dict[field.name] = _mark_int64_fields_for_proto_maps(value, field.message_type.fields_by_name['value'].type)
            continue
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            ftype = partial(_mark_int64_fields)
        elif field.type in _PROTOBUF_INT64_FIELDS:
            ftype = int
        else:
            continue
        json_dict[field.name] = [ftype(v) for v in value] if field.label == FieldDescriptor.LABEL_REPEATED else ftype(value)
    return json_dict