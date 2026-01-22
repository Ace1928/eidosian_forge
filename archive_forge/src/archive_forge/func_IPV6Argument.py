from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def IPV6Argument(value):
    """Argparse argument type that checks for a valid ipv6 address."""
    if not IsValidIPV6(value):
        raise argparse.ArgumentTypeError("invalid ipv6 value: '{0}'".format(value))
    return value