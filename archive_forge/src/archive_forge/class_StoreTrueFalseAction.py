import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
class StoreTrueFalseAction(argparse._StoreTrueAction):
    """Argparse action that acts as a combination of store_true and store_false.

  Calliope already gives any bool-type arguments the standard and `--no-`
  variants. In most cases we only want to document the option that does
  something---if we have `default=False`, we don't want to show `--no-foo`,
  since it won't do anything.

  But in some cases we *do* want to show both variants: one example is when
  `--foo` means "enable," `--no-foo` means "disable," and neither means "do
  nothing." The obvious way to represent this is `default=None`; however, (1)
  the default value of `default` is already None, so most boolean actions would
  have this setting by default (not what we want), and (2) we still want an
  option to have this True/False/None behavior *without* the flag documentation.

  To get around this, we have an opt-in version of the same thing that documents
  both the flag and its inverse.
  """

    def __init__(self, *args, **kwargs):
        super(StoreTrueFalseAction, self).__init__(*args, default=None, **kwargs)