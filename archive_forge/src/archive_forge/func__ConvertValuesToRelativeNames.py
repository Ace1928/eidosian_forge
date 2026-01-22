from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def _ConvertValuesToRelativeNames(names, resource_parser):
    if names:
        names = [resource_parser(name).RelativeName() for name in names]
    return names