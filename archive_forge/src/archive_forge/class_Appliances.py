from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Appliances(base.Group):
    """Manage Transfer Appliances.

  Transfer Appliances are high-capacity storage devices that enable
  the transfer and secure shipment of data to a Google upload facility, where
  data is uploaded to Cloud Storage.
  """