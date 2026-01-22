from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PatientScope(_messages.Message):
    """Apply consents given by a list of patients.

  Fields:
    patientIds: Optional. The list of patient IDs whose Consent resources will
      be enforced. At most 10,000 patients can be specified. An empty list is
      equivalent to all patients (meaning the entire FHIR store).
  """
    patientIds = _messages.StringField(1, repeated=True)