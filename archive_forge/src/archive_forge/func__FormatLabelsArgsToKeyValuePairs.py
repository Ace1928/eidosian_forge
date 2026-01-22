from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def _FormatLabelsArgsToKeyValuePairs(labels):
    """Flattens the labels specified in cli to a list of (k, v) pairs."""
    labels = [] if labels is None else labels
    labels_flattened = []
    for labels_sublist in labels:
        labels_flattened.extend([label.strip() for label in labels_sublist])
    labels_flattened_unique = list(set(labels_flattened))
    return [_ExtractKeyValueFromLabel(label) for label in labels_flattened_unique]