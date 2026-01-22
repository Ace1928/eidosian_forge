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
def GenerateChoices(field, attributes):
    variant = field.variant if field else None
    choices = None
    if attributes.choices is not None:
        choice_map = {c.arg_value: c.help_text for c in attributes.choices}
        choices = choice_map if any(choice_map.values()) else sorted(choice_map.keys())
    elif variant == messages.Variant.ENUM:
        choices = [EnumNameToChoice(name) for name in sorted(field.type.names())]
    return choices