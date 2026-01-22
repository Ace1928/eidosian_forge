from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ScopeStatesValue(_messages.Message):
    """Output only. Scope-specific Feature status. If this Feature does
    report any per-Scope status, this field may be unused. The keys indicate
    which Scope the state is for, in the form:
    `projects/{p}/locations/global/scopes/{s}` Where {p} is the project, {s}
    is a valid Scope in this project. {p} WILL match the Feature's project.

    Messages:
      AdditionalProperty: An additional property for a ScopeStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ScopeStatesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ScopeStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A ScopeFeatureState attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ScopeFeatureState', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)