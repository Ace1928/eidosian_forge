import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdditionalPropertiesWithEnumMessage(messages.Message):

    class AdditionalProperty(messages.Message):
        key = messages.StringField(1)
        value = messages.MessageField(MessageWithEnum, 2)
    additionalProperties = messages.MessageField('AdditionalProperty', 1, repeated=True)