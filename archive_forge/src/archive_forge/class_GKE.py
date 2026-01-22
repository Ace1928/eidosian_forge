from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GKE(_messages.Message):
    """Represents a GKE destination.

  Fields:
    cluster: Required. The name of the cluster the GKE service is running in.
      The cluster must be running in the same project as the trigger being
      created.
    location: Required. The name of the Google Compute Engine in which the
      cluster resides, which can either be compute zone (for example, us-
      central1-a) for the zonal clusters or region (for example, us-central1)
      for regional clusters.
    namespace: Required. The namespace the GKE service is running in.
    path: Optional. The relative path on the GKE service the events should be
      sent to. The value must conform to the definition of a URI path segment
      (section 3.3 of RFC2396). Examples: "/route", "route", "route/subroute".
    service: Required. Name of the GKE service.
  """
    cluster = _messages.StringField(1)
    location = _messages.StringField(2)
    namespace = _messages.StringField(3)
    path = _messages.StringField(4)
    service = _messages.StringField(5)