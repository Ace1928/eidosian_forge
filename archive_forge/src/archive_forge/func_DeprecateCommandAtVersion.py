from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def DeprecateCommandAtVersion(remove_version, remove=False, alt_command=None):
    """Decorator that marks a GCloud command as deprecated.

  Args:
      remove_version: string, The GCloud sdk version where this command will be
      marked as removed.

      remove: boolean, True if the command should be removed in underlying
      base.Deprecate decorator, False if it should only print a warning

      alt_command: string, optional alternative command to use in place of
      deprecated command

  Raises:
      ValueError: If remove version is missing

  Returns:
    A modified version of the provided class.
  """
    if not remove_version:
        raise ValueError('Valid remove version is required')
    warn = _WARNING_MSG.format(remove_version)
    error = _REMOVED_MSG.format(remove_version)
    if alt_command:
        warn += _COMMAND_ALT_MSG.format(alt_command)
        error += _COMMAND_ALT_MSG.format(alt_command)
    return base.Deprecate(is_removed=remove, warning=warn, error=error)