from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SslCertificates(base.Group):
    """View and manage your App Engine SSL certificates.

  This set of commands can be used to view and manage your app's
  SSL certificates.
  """
    category = base.APP_ENGINE_CATEGORY
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To list your App Engine SSL certificates, run:\n\n            $ {command} list\n      '}