from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FhirStoreMetrics(_messages.Message):
    """List of metrics for a given FHIR store.

  Fields:
    metrics: List of FhirStoreMetric by resource type.
    name: The resource name of the FHIR store to get metrics for, in the
      format `projects/{project_id}/datasets/{dataset_id}/fhirStores/{fhir_sto
      re_id}`.
  """
    metrics = _messages.MessageField('FhirStoreMetric', 1, repeated=True)
    name = _messages.StringField(2)