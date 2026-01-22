from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.core import exceptions
import six
def _GetUnrecognizedFieldPaths(message):
    """Returns the field paths for unrecognized fields in the message."""
    errors = encoding.UnrecognizedFieldIter(message)
    unrecognized_field_paths = []
    for edges_to_message, field_names in errors:
        message_field_path = '.'.join((six.text_type(e) for e in edges_to_message))
        for field_name in field_names:
            unrecognized_field_paths.append('{}.{}'.format(message_field_path, field_name))
    return sorted(unrecognized_field_paths)