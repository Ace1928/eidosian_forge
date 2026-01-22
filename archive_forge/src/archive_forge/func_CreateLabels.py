from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def CreateLabels(args, messages):
    """Parses and validates labels input."""
    labels_dict = yaml.load(args.filter_labels)
    if len(labels_dict) > 1:
        raise InvalidLabelInput('The input is not valid for `--filter-labels`. It must be one key/value pair.')
    key = list(labels_dict.keys())[0]
    if len(labels_dict[key]) > 1:
        raise InvalidLabelInput('The input is not valid for `--filter-labels`. It must be one key with one value.')
    value = labels_dict[key][0]
    return messages.LabelsValue.AdditionalProperty(key=key, value=[extra_types.JsonValue(string_value=value)])