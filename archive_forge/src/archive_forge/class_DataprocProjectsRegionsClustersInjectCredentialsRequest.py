from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersInjectCredentialsRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersInjectCredentialsRequest object.

  Fields:
    cluster: Required. The cluster, in the form clusters/.
    injectCredentialsRequest: A InjectCredentialsRequest resource to be passed
      as the request body.
    project: Required. The ID of the Google Cloud Platform project the cluster
      belongs to, of the form projects/.
    region: Required. The region containing the cluster, of the form regions/.
  """
    cluster = _messages.StringField(1, required=True)
    injectCredentialsRequest = _messages.MessageField('InjectCredentialsRequest', 2)
    project = _messages.StringField(3, required=True)
    region = _messages.StringField(4, required=True)