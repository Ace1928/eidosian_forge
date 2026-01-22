from __future__ import absolute_import
import re
def _GetHandlerType(handler):
    """Get handler type of mapping.

  Args:
    handler: Original handler.

  Returns:
    Handler type determined by which handler id attribute is set.

  Raises:
    ValueError: when none of the handler id attributes are set.
  """
    if 'apiEndpoint' in handler:
        return 'apiEndpoint'
    elif 'staticDir' in handler:
        return 'staticDirectory'
    elif 'path' in handler:
        return 'staticFiles'
    elif 'scriptPath' in handler:
        return 'script'
    raise ValueError('Unrecognized handler type: %s' % handler)