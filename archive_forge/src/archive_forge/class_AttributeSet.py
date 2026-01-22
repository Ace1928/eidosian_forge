from decimal import Decimal
from boto.compat import filter, map
class AttributeSet(ResponseElement):
    ItemDimensions = Element(ComplexDimensions)
    ListPrice = Element(ComplexMoney)
    PackageDimensions = Element(ComplexDimensions)
    SmallImage = Element(Image)