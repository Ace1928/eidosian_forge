from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def TimeoutFlag():
    return StringFlag('--timeout', help='Maximum request execution time (timeout). It is specified as a duration; for example, "10m5s" is ten minutes and five seconds. If you don\'t specify a unit, seconds is assumed. For example, "10" is 10 seconds. Specify "0" to set the timeout to the default value.', type=arg_parsers.Duration(lower_bound='0s', parsed_unit='s'))