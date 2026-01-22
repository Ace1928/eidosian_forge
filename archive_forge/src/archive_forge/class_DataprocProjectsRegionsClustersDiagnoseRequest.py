from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersDiagnoseRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersDiagnoseRequest object.

  Fields:
    clusterName: Required. The cluster name.
    diagnoseClusterRequest: A DiagnoseClusterRequest resource to be passed as
      the request body.
    projectId: Required. The ID of the Google Cloud Platform project that the
      cluster belongs to.
    region: Required. The Dataproc region in which to handle the request.
  """
    clusterName = _messages.StringField(1, required=True)
    diagnoseClusterRequest = _messages.MessageField('DiagnoseClusterRequest', 2)
    projectId = _messages.StringField(3, required=True)
    region = _messages.StringField(4, required=True)