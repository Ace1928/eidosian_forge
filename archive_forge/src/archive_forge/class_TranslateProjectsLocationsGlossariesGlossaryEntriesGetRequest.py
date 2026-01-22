from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesGlossaryEntriesGetRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesGlossaryEntriesGetRequest object.

  Fields:
    name: Required. The resource name of the glossary entry to get
  """
    name = _messages.StringField(1, required=True)