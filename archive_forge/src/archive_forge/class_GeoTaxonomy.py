from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GeoTaxonomy(_messages.Message):
    """Encapsulates the geographic taxonomy data for a sku.

  Enums:
    TypeValueValuesEnum: The type of Geo Taxonomy: GLOBAL, REGIONAL, or
      MULTI_REGIONAL.

  Fields:
    regions: The list of regions associated with a sku. Empty for Global skus,
      which are associated with all Google Cloud regions.
    type: The type of Geo Taxonomy: GLOBAL, REGIONAL, or MULTI_REGIONAL.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of Geo Taxonomy: GLOBAL, REGIONAL, or MULTI_REGIONAL.

    Values:
      TYPE_UNSPECIFIED: The type is not specified.
      GLOBAL: The sku is global in nature, e.g. a license sku. Global skus are
        available in all regions, and so have an empty region list.
      REGIONAL: The sku is available in a specific region, e.g. "us-west2".
      MULTI_REGIONAL: The sku is associated with multiple regions, e.g. "us-
        west2" and "us-east1".
    """
        TYPE_UNSPECIFIED = 0
        GLOBAL = 1
        REGIONAL = 2
        MULTI_REGIONAL = 3
    regions = _messages.StringField(1, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 2)