from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iap import util as iap_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetALPHA(Set):
    """Set the setting for an IAP resource."""

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        iap_util.AddIapSettingArg(parser, use_region_arg=True)
        iap_util.AddIapSettingFileArg(parser)
        base.URI_FLAG.RemoveFromParser(parser)