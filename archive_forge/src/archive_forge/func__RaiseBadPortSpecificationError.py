from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def _RaiseBadPortSpecificationError(invalid_flag, port_spec_flag, invalid_value):
    raise calliope_exceptions.InvalidArgumentException(port_spec_flag, '{0} cannot be specified when using: {1}.'.format(invalid_flag, invalid_value))