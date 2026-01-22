from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FamilySubsetsValue(_messages.Message):
    """Map from column family name to the columns in this family to be
    included in the AuthorizedView.

    Messages:
      AdditionalProperty: An additional property for a FamilySubsetsValue
        object.

    Fields:
      additionalProperties: Additional properties of type FamilySubsetsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FamilySubsetsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleBigtableAdminV2AuthorizedViewFamilySubsets attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleBigtableAdminV2AuthorizedViewFamilySubsets', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)