from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
@staticmethod
def AllValues():
    """Gets all possible enum values.

    Returns:
      list, All the enum values.
    """
    return list(Architecture._ALL)