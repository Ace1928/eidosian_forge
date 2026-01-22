from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsGetSupportedLanguagesRequest(_messages.Message):
    """A TranslateProjectsGetSupportedLanguagesRequest object.

  Fields:
    displayLanguageCode: Optional. The language to use to return localized,
      human readable names of supported languages. If missing, then display
      names are not returned in a response.
    model: Optional. Get supported languages of this model. The format depends
      on model type: - AutoML Translation models: `projects/{project-number-
      or-id}/locations/{location-id}/models/{model-id}` - General (built-in)
      models: `projects/{project-number-or-id}/locations/{location-
      id}/models/general/nmt`, Returns languages supported by the specified
      model. If missing, we get supported languages of Google general NMT
      model.
    parent: Required. Project or location to make a call. Must refer to a
      caller's project. Format: `projects/{project-number-or-id}` or
      `projects/{project-number-or-id}/locations/{location-id}`. For global
      calls, use `projects/{project-number-or-id}/locations/global` or
      `projects/{project-number-or-id}`. Non-global location is required for
      AutoML models. Only models within the same region (have same location-
      id) can be used, otherwise an INVALID_ARGUMENT (400) error is returned.
  """
    displayLanguageCode = _messages.StringField(1)
    model = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)