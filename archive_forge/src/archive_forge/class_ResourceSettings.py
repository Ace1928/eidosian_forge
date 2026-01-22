from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class ResourceSettings(base.Group):
    """Create and manage Resource Settings.

  The gcloud Resource Settings command group lets you create and manipulate
  resource settings.

  The Resource Settings Service is a hierarchy-aware service with a
  public-facing API for users to store settings that modify the behavior
  of their Google Cloud Platform resources, such as virtual machines,
  firewalls, projects, and so forth. Settings can be attached to
  organizations, folders, and projects, and can influence these resources
  as well as service resources that are descendants of the resource to which
  the settings are attached.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.EnableUserProjectQuotaWithFallback()