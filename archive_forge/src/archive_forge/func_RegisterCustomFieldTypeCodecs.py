from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import os
import re
import textwrap
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding as api_encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import encoding
def RegisterCustomFieldTypeCodecs(field_type_codecs):
    """Registers custom field codec for int64s."""

    def _EncodeInt64Field(unused_field, value):
        int_value = api_encoding.CodecResult(value=value, complete=True)
        return int_value

    def _DecodeInt64Field(unused_field, value):
        return api_encoding.CodecResult(value=value, complete=True)
    field_type_codecs[messages.IntegerField] = encoding_helper._Codec(encoder=_EncodeInt64Field, decoder=_DecodeInt64Field)
    return field_type_codecs