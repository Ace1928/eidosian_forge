from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesPatchRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesPatchRequest object.

  Fields:
    glossary: A Glossary resource to be passed as the request body.
    name: Required. The resource name of the glossary. Glossary names have the
      form `projects/{project-number-or-id}/locations/{location-
      id}/glossaries/{glossary-id}`.
    updateMask: The list of fields to be updated. Currently only
      `display_name` and 'input_config'
  """
    glossary = _messages.MessageField('Glossary', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)