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
def Action(self):
    """The argparse action to use for this argument.

    'store' is the default action, but sometimes something like 'append' might
    be required to allow the argument to be repeated and all values collected.

    Returns:
      str, The argparse action to use.
    """
    return 'store'