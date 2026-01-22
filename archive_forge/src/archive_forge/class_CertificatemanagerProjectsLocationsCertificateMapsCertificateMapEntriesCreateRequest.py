from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntriesCreateRequest(_messages.Message):
    """A CertificatemanagerProjectsLocationsCertificateMapsCertificateMapEntrie
  sCreateRequest object.

  Fields:
    certificateMapEntry: A CertificateMapEntry resource to be passed as the
      request body.
    certificateMapEntryId: Required. A user-provided name of the certificate
      map entry.
    parent: Required. The parent resource of the certificate map entry. Must
      be in the format `projects/*/locations/*/certificateMaps/*`.
  """
    certificateMapEntry = _messages.MessageField('CertificateMapEntry', 1)
    certificateMapEntryId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)