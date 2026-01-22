from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesGlossaryEntriesDeleteRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesGlossaryEntriesDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the glossary entry to delete
  """
    name = _messages.StringField(1, required=True)