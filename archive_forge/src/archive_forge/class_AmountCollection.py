from decimal import Decimal
from boto.compat import filter, map
class AmountCollection(ResponseElement):

    def startElement(self, name, attrs, connection):
        setattr(self, name, ComplexAmount(name=name))
        return getattr(self, name)