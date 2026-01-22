import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def DictToAdditionalPropertyMessage(properties, additional_property_type, sort_items=False):
    """Convert the given dictionary to an AdditionalProperty message."""
    items = properties.items()
    if sort_items:
        items = sorted(items)
    map_ = []
    for key, value in items:
        map_.append(additional_property_type.AdditionalProperty(key=key, value=value))
    return additional_property_type(additionalProperties=map_)