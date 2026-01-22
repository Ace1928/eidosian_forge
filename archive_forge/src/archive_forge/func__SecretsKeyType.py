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
def _SecretsKeyType(key):
    """Validates and canonicalizes secrets key configuration.

  Args:
    key: Secrets key configuration.

  Returns:
    Canonicalized secrets key configuration.

  Raises:
    ArgumentTypeError: Secrets key configuration is not valid.
  """
    if not key.strip():
        raise ArgumentTypeError('Secret environment variable names/secret paths cannot be empty.')
    canonicalized_key = key
    if _SECRET_PATH_PATTERN.search(key):
        canonicalized_key = _CanonicalizePath(key)
    else:
        if '/' in key:
            log.warning("'{}' will be interpreted as a secret environment variable name as it doesn't match the pattern for a secret path '/mount_path:/secret_file_path'.".format(key))
        if key.startswith('X_GOOGLE_') or key in ['GOOGLE_ENTRYPOINT', 'GOOGLE_FUNCTION_TARGET', 'GOOGLE_RUNTIME', 'GOOGLE_RUNTIME_VERSION']:
            raise ArgumentTypeError("Secret environment variable name '{}' is reserved for internal use.".format(key))
    return canonicalized_key