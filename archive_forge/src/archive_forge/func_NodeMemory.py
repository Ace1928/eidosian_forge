from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
def NodeMemory(value):
    """Declarative command argument type for node-memory flag."""
    size = arg_parsers.BinarySize(suggested_binary_size_scales=['MB', 'GB'], default_unit='MB')
    return int(size(value) / 1024 / 1024)