from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def GetTermSize():
    """Gets the terminal x and y dimensions in characters.

  _GetTermSize*() helper functions taken from:
    http://stackoverflow.com/questions/263890/

  Returns:
    (columns, lines): A tuple containing the terminal x and y dimensions.
  """
    xy = None
    for get_terminal_size in (_GetTermSizePosix, _GetTermSizeWindows, _GetTermSizeEnvironment, _GetTermSizeTput):
        try:
            xy = get_terminal_size()
            if xy:
                break
        except:
            pass
    return xy or (80, 24)