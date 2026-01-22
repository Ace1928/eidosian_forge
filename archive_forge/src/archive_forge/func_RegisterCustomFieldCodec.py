import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def RegisterCustomFieldCodec(encoder, decoder):
    """Register a custom encoder/decoder for this field."""

    def Register(field):
        _CUSTOM_FIELD_CODECS[field] = _Codec(encoder=encoder, decoder=decoder)
        return field
    return Register