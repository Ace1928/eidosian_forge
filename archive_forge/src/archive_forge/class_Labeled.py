from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
class Labeled(Table):
    """Marker class for a list of "Label: value" 2-tuples."""
    skip_empty = True
    separator = ':'