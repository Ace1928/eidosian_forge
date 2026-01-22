from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def GetCreateLabelsFlag(extra_message='', labels_name='labels', validate_keys=True, validate_values=True):
    """Makes the base.Argument for --labels flag."""
    key_type = KEY_FORMAT_VALIDATOR if validate_keys else None
    value_type = VALUE_FORMAT_VALIDATOR if validate_values else None
    format_help = []
    if validate_keys:
        format_help.append(KEY_FORMAT_HELP)
    if validate_values:
        format_help.append(VALUE_FORMAT_HELP)
    help_parts = ['List of label KEY=VALUE pairs to add.']
    if format_help:
        help_parts.append(' '.join(format_help))
    if extra_message:
        help_parts.append(extra_message)
    return base.Argument('--{}'.format(labels_name), metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=key_type, value_type=value_type), action=arg_parsers.UpdateAction, help='\n\n'.join(help_parts))