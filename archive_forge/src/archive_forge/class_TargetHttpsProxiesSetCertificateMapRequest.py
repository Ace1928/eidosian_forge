from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetHttpsProxiesSetCertificateMapRequest(_messages.Message):
    """A TargetHttpsProxiesSetCertificateMapRequest object.

  Fields:
    certificateMap: URL of the Certificate Map to associate with this
      TargetHttpsProxy. Accepted format is
      //certificatemanager.googleapis.com/projects/{project
      }/locations/{location}/certificateMaps/{resourceName}.
  """
    certificateMap = _messages.StringField(1)