from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.core.util import encoding
def GetDecodedArgv():
    return [encoding.Decode(a) for a in sys.argv]