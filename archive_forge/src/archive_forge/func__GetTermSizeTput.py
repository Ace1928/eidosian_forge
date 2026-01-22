from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetTermSizeTput():
    """Returns the terminal x and y dimemsions from tput(1)."""
    import subprocess
    output = encoding.Decode(subprocess.check_output(['tput', 'cols'], stderr=subprocess.STDOUT))
    cols = int(output)
    output = encoding.Decode(subprocess.check_output(['tput', 'lines'], stderr=subprocess.STDOUT))
    rows = int(output)
    return (cols, rows)