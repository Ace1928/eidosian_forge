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
class UpdateAction(argparse.Action):
    """Create a single dict value from delimited or repeated flags.

  This class is intended to be a more flexible version of
  argparse._AppendAction.

  For example, with the following flag definition:

      parser.add_argument(
        '--inputs',
        type=arg_parsers.ArgDict(),
        action='append')

  a caller can specify on the command line flags such as:

    --inputs k1=v1,k2=v2

  and the result will be a list of one dict:

    [{ 'k1': 'v1', 'k2': 'v2' }]

  Specifying two separate command line flags such as:

    --inputs k1=v1 \\
    --inputs k2=v2

  will produce a list of dicts:

    [{ 'k1': 'v1'}, { 'k2': 'v2' }]

  The UpdateAction class allows for both of the above user inputs to result
  in the same: a single dictionary:

    { 'k1': 'v1', 'k2': 'v2' }

  This gives end-users a lot more flexibility in constructing their command
  lines, especially when scripting calls.

  Note that this class will raise an exception if a key value is specified
  more than once. To allow for a key value to be specified multiple times,
  use UpdateActionWithAppend.
  """

    def OnDuplicateKeyRaiseError(self, key, existing_value=None, new_value=None):
        if existing_value is None:
            user_input = None
        else:
            user_input = ', '.join([six.text_type(existing_value), six.text_type(new_value)])
        raise argparse.ArgumentError(self, _GenerateErrorMessage('"{0}" cannot be specified multiple times'.format(key), user_input=user_input))

    def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, onduplicatekey_handler=OnDuplicateKeyRaiseError):
        if nargs == 0:
            raise ValueError('nargs for append actions must be > 0; if arg strings are not supplying the value to append, the append const action may be more appropriate')
        if const is not None and nargs != argparse.OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % argparse.OPTIONAL)
        self.choices = choices
        if isinstance(choices, dict):
            choices = sorted(choices.keys())
        super(UpdateAction, self).__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default, type=type, choices=choices, required=required, help=help, metavar=metavar)
        self.onduplicatekey_handler = onduplicatekey_handler

    def _EnsureValue(self, namespace, name, value):
        if getattr(namespace, name, None) is None:
            setattr(namespace, name, value)
        return getattr(namespace, name)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, dict):
            items = copy.copy(self._EnsureValue(namespace, self.dest, collections.OrderedDict()))
            for k, v in six.iteritems(values):
                if k in items:
                    v = self.onduplicatekey_handler(self, k, items[k], v)
                items[k] = v
        else:
            items = copy.copy(self._EnsureValue(namespace, self.dest, []))
            for k in values:
                if k in items:
                    self.onduplicatekey_handler(self, k)
                else:
                    items.append(k)
        setattr(namespace, self.dest, items)