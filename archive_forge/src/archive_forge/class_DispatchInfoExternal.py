from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class DispatchInfoExternal(validation.Validated):
    """Describes the format of a dispatch.yaml file."""
    ATTRIBUTES = {APPLICATION: validation.Optional(appinfo.APPLICATION_RE_STRING), DISPATCH: validation.Optional(validation.Repeated(DispatchEntry))}