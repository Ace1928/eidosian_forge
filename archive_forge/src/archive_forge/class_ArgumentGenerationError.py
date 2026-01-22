from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import re
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import http_encoding
import six
class ArgumentGenerationError(Error):
    """Generic error when we can't auto generate an argument for an api field."""

    def __init__(self, field_name, reason):
        super(ArgumentGenerationError, self).__init__('Failed to generate argument for field [{}]: {}'.format(field_name, reason))