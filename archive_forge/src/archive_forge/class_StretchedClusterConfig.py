from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StretchedClusterConfig(_messages.Message):
    """Configuration of a stretched cluster.

  Fields:
    preferredLocation: Required. Zone that will remain operational when
      connection between the two zones is lost. Specify the resource name of a
      zone that belongs to the region of the private cloud. For example:
      `projects/{project}/locations/europe-west3-a` where `{project}` can
      either be a project number or a project ID.
    secondaryLocation: Required. Additional zone for a higher level of
      availability and load balancing. Specify the resource name of a zone
      that belongs to the region of the private cloud. For example:
      `projects/{project}/locations/europe-west3-b` where `{project}` can
      either be a project number or a project ID.
  """
    preferredLocation = _messages.StringField(1)
    secondaryLocation = _messages.StringField(2)