import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def DoRoundtrip(class_type, json_msg=None, message=None, times=4):
    if json_msg:
        json_msg = MtoJ(JtoM(class_type, json_msg))
    if message:
        message = JtoM(class_type, MtoJ(message))
    if times == 0:
        result = json_msg if json_msg else message
        return result
    return DoRoundtrip(class_type=class_type, json_msg=json_msg, message=message, times=times - 1)