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
def CheckValidEnumNames(api_names, choices_values):
    """Ensures the api_name given in the spec matches a value from the API."""
    if api_names:
        bad_choices = [name for name in choices_values if not (name in api_names or ChoiceToEnumName(six.text_type(name)) in api_names)]
    else:
        bad_choices = []
    if bad_choices:
        raise arg_parsers.ArgumentTypeError('{} is/are not valid enum values.'.format(', '.join(bad_choices)))