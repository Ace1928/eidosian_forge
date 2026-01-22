from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddCreateLabelsFlag(parser, labels_name, resource_name, extra_message='', validate_values=True, skip_extra_message=False):
    """Add create labels flags."""
    if not skip_extra_message:
        extra_message += 'If the {0} was given as a JSON/YAML object from a string or file, this flag will replace the labels value in the given {0}.'.format(resource_name)
    labels_util.GetCreateLabelsFlag(extra_message=extra_message, labels_name=labels_name, validate_values=validate_values).AddToParser(parser)