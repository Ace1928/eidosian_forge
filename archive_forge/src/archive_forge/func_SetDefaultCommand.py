from mx import DateTime
from __future__ import absolute_import
from __future__ import print_function
import os
import pdb
import sys
import traceback
from absl import app
from absl import flags
def SetDefaultCommand(default_command):
    """Change the default command to execute if none is explicitly given.

  Args:
    default_command: str, the name of the command to execute by default.
  """
    global _cmd_default
    _cmd_default = default_command