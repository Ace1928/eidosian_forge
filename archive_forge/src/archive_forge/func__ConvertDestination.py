from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.edgeml import operations
from googlecloudsdk.api_lib.edgeml import util
from googlecloudsdk.core import properties
def _ConvertDestination(source):
    """Converts model[/saved_model.(pb|pbtxt)] style filename to model.tflite."""
    return re.sub('(/saved_model\\.(pb|pbtxt))?$', '.tflite', source, count=1)