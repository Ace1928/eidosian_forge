from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Recaptcha(base.Group):
    """Manage reCAPTCHA Enterprise Keys.

  Commands for managing reCAPTCHA Enterprise Keys.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        del context, args