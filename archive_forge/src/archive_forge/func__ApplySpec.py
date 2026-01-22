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
def _ApplySpec(self, key, value):
    if key in self.spec:
        if self.spec[key] is None:
            if value:
                raise ArgumentTypeError('Key [{0}] does not take a value'.format(key))
            return None
        return self.spec[key](value)
    else:
        raise ArgumentTypeError(_GenerateErrorMessage('valid keys are [{0}]'.format(', '.join(sorted(self.spec.keys()))), user_input=key))