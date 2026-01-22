from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateIssuanceConfigsListRequest(_messages.Message):
    """A
  CertificatemanagerProjectsLocationsCertificateIssuanceConfigsListRequest
  object.

  Fields:
    filter: Filter expression to restrict the Certificates Configs returned.
    orderBy: A list of Certificate Config field names used to specify the
      order of the returned results. The default sorting order is ascending.
      To specify descending order for a field, add a suffix `" desc"`.
    pageSize: Maximum number of certificate configs to return per call.
    pageToken: The value returned by the last
      `ListCertificateIssuanceConfigsResponse`. Indicates that this is a
      continuation of a prior `ListCertificateIssuanceConfigs` call, and that
      the system should return the next page of data.
    parent: Required. The project and location from which the certificate
      should be listed, specified in the format `projects/*/locations/*`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)