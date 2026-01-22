from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class InvalidResourceArgumentLists(Error):
    """Exception for missing, extra, or out of order arguments."""

    def __init__(self, expected, actual):
        expected = ['[' + e + ']' if e in IGNORED_FIELDS else e for e in expected]
        super(InvalidResourceArgumentLists, self).__init__('Invalid resource arguments: Expected [{}], Found [{}].'.format(', '.join(expected), ', '.join(actual)))