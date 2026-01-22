import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def IndentKwargs(kwargs):
    kwargs = dict(kwargs)
    kwargs['indent'] = kwargs.get('indent', 0) + 4
    return kwargs