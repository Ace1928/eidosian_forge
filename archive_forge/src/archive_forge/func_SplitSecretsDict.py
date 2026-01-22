from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def SplitSecretsDict(secrets_dict):
    """Splits the secrets dict into sorted ordered dicts for each secret type.

  Args:
    secrets_dict: Secrets configuration dict.

  Returns:
    A tuple (secret_env_vars, secret_volumes) of sorted ordered dicts for each
    secret type.
  """
    secret_volumes = {k: v for k, v in six.iteritems(secrets_dict) if _SECRET_PATH_PATTERN.search(k)}
    secret_env_vars = {k: v for k, v in six.iteritems(secrets_dict) if k not in secret_volumes}
    return (collections.OrderedDict(sorted(six.iteritems(secret_env_vars))), collections.OrderedDict(sorted(six.iteritems(secret_volumes))))