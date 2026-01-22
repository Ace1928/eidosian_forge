import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def RegisterCustomMessageCodec(encoder, decoder):
    """Register a custom encoder/decoder for this message class."""

    def Register(cls):
        _CUSTOM_MESSAGE_CODECS[cls] = _Codec(encoder=encoder, decoder=decoder)
        return cls
    return Register