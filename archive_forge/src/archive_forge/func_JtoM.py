import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def JtoM(class_type, json_str):
    return encoding.JsonToMessage(class_type, json_str)