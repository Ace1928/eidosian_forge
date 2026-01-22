from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ContactsPerson(_messages.Message):
    """A contact person for the entry.

  Fields:
    designation: Designation of the person, for example, Data Steward.
    email: Email of the person in the format of `john.doe@xyz`, ``, or `John
      Doe`.
  """
    designation = _messages.StringField(1)
    email = _messages.StringField(2)