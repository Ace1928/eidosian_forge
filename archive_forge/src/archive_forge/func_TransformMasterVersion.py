from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core.util import times
def TransformMasterVersion(r, undefined=''):
    """Returns the formatted master version.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.
  Returns:
    The formatted master version.
  """
    version = r.get('currentMasterVersion', None)
    if version is None:
        return undefined
    if r.get('enableKubernetesAlpha', False):
        version = '{0} ALPHA'.format(version)
    try:
        time_left = ParseExpireTime(r.get('expireTime', None))
        if time_left is not None:
            if time_left.days > constants.EXPIRE_WARNING_DAYS:
                version += ' ({0} days left)'.format(time_left.days)
            else:
                version += ' (! {0} days left !)'.format(time_left.days)
        return version
    except times.Error:
        return undefined
    return version