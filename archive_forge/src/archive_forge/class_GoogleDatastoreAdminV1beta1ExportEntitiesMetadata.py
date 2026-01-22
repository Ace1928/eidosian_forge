from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1beta1ExportEntitiesMetadata(_messages.Message):
    """Metadata for ExportEntities operations.

  Fields:
    common: Metadata common to all Datastore Admin operations.
    entityFilter: Description of which entities are being exported.
    outputUrlPrefix: Location for the export metadata and data files. This
      will be the same value as the
      google.datastore.admin.v1beta1.ExportEntitiesRequest.output_url_prefix
      field. The final output location is provided in
      google.datastore.admin.v1beta1.ExportEntitiesResponse.output_url.
    progressBytes: An estimate of the number of bytes processed.
    progressEntities: An estimate of the number of entities processed.
  """
    common = _messages.MessageField('GoogleDatastoreAdminV1beta1CommonMetadata', 1)
    entityFilter = _messages.MessageField('GoogleDatastoreAdminV1beta1EntityFilter', 2)
    outputUrlPrefix = _messages.StringField(3)
    progressBytes = _messages.MessageField('GoogleDatastoreAdminV1beta1Progress', 4)
    progressEntities = _messages.MessageField('GoogleDatastoreAdminV1beta1Progress', 5)