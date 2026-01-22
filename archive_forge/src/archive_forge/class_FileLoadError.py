from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml_location_value
from googlecloudsdk.core.util import files
from ruamel import yaml
import six
class FileLoadError(Error):
    """An error that wraps errors when loading/reading files."""

    def __init__(self, e, f):
        super(FileLoadError, self).__init__(e, verb='load', f=f)