from decimal import Decimal
from boto.compat import filter, map
class VariationRelationship(ResponseElement):
    Identifiers = Element(MarketplaceASIN=Element(), SKUIdentifier=Element())
    GemType = SimpleList()
    MaterialType = SimpleList()
    OperatingSystem = SimpleList()