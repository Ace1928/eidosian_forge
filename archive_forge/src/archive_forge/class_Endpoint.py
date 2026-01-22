from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Endpoint(base.Group):
    """Manage Vertex AI endpoints.

     An endpoint contains one or more deployed models, all of which must have
     the same interface but may come from different models.
     An endpoint is to obtain online prediction and explanation from one of
     its deployed models.

     When you communicate with Vertex AI services, you identify a specific
     endpoint that is deployed in the cloud using a combination of the current
     project, the region, and the endpoint.
  """
    category = base.VERTEX_AI_CATEGORY