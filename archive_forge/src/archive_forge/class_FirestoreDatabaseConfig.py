from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreDatabaseConfig(_messages.Message):
    """Message for defining firestore database configuration.

  Fields:
    location_id: LocationID is the location of the database. Available
      databases locations are listed at
      https://cloud.google.com/firestore/docs/locations.
  """
    location_id = _messages.StringField(1)