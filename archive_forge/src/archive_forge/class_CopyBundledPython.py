from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.components import util
from googlecloudsdk.core.updater import update_manager
@base.Hidden
class CopyBundledPython(base.Command):
    """Make a temporary copy of bundled Python installation.

  Also print its location.

  If the Python installation used to execute this command is *not* bundled, do
  not make a copy. Instead, print the location of the current Python
  installation.
  """

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('value(python_location)')

    def Run(self, args):
        manager = util.GetUpdateManager(args)
        if manager.IsPythonBundled():
            python_location = update_manager.CopyPython()
        else:
            python_location = sys.executable
        return {'python_location': python_location}