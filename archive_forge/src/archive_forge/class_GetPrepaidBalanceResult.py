from decimal import Decimal
from boto.compat import filter, map
class GetPrepaidBalanceResult(ResponseElement):

    def startElement(self, name, attrs, connection):
        if name == 'PrepaidBalance':
            setattr(self, name, AmountCollection(name=name))
            return getattr(self, name)
        return super(GetPrepaidBalanceResult, self).startElement(name, attrs, connection)