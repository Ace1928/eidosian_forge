from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
class YAMLData(object):
    """A general data holder object for data parsed from a YAML file."""

    def __init__(self, data):
        self._data = data

    def GetData(self):
        return self._data