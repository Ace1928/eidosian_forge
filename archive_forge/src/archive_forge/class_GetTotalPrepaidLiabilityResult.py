from decimal import Decimal
from boto.compat import filter, map
class GetTotalPrepaidLiabilityResult(ResponseElement):

    def startElement(self, name, attrs, connection):
        if name == 'OutstandingPrepaidLiability':
            setattr(self, name, AmountCollection(name=name))
            return getattr(self, name)
        return super(GetTotalPrepaidLiabilityResult, self).startElement(name, attrs, connection)