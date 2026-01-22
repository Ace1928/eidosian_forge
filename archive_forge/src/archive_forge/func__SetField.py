import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def _SetField(dictblob, field_path, value):
    for field in field_path[:-1]:
        dictblob = dictblob.setdefault(field, {})
    dictblob[field_path[-1]] = value