from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import firestore_util
from googlecloudsdk.command_lib.emulators import flags
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.command_lib.util import java
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Firestore(base.Group):
    """Manage your local Firestore emulator.

  This set of commands allows you to start and use a local Firestore emulator.
  """
    detailed_help = {'EXAMPLES': '          To start the local Firestore emulator, run:\n\n            $ {command} start\n          '}

    def Filter(self, context, args):
        java.RequireJavaInstalled(firestore_util.FIRESTORE_TITLE, min_version=8)
        util.EnsureComponentIsInstalled('cloud-firestore-emulator', firestore_util.FIRESTORE_TITLE)