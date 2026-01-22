from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesGetRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesGetRequest object.

  Fields:
    name: Required. The name of the glossary to retrieve.
  """
    name = _messages.StringField(1, required=True)