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
def _DecodeDateTimeField(unused_field, value):
    result = protojson.ProtoJson().decode_field(message_types.DateTimeField(1), value)
    return encoding.CodecResult(value=result, complete=True)