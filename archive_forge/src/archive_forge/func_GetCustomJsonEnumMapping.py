import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def GetCustomJsonEnumMapping(enum_type, python_name=None, json_name=None):
    """Return the appropriate remapping for the given enum, or None."""
    return _FetchRemapping(enum_type, 'enum', python_name=python_name, json_name=json_name, mappings=_JSON_ENUM_MAPPINGS)