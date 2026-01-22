import datetime
import json
import numbers
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def _JsonArrayToPythonValue(json_value):
    util.Typecheck(json_value, JsonArray)
    return [_JsonValueToPythonValue(e) for e in json_value.entries]