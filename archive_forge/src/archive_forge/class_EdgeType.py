import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
class EdgeType(object):
    """The type of transition made by an edge."""
    SCALAR = 1
    REPEATED = 2
    MAP = 3