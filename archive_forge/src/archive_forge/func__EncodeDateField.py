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
def _EncodeDateField(field, value):
    """Encoder for datetime.date objects."""
    if field.repeated:
        result = [d.isoformat() for d in value]
    else:
        result = value.isoformat()
    return encoding.CodecResult(value=result, complete=True)