from decimal import Decimal
from boto.compat import filter, map
class ComplexAmount(ResponseElement):

    def __repr__(self):
        return '{0} {1}'.format(self.CurrencyCode, self.Value)

    def __float__(self):
        return float(self.Value)

    def __str__(self):
        return str(self.Value)

    def startElement(self, name, attrs, connection):
        if name not in ('CurrencyCode', 'Value'):
            message = 'Unrecognized tag {0} in ComplexAmount'.format(name)
            raise AssertionError(message)
        return super(ComplexAmount, self).startElement(name, attrs, connection)

    def endElement(self, name, value, connection):
        if name == 'Value':
            value = Decimal(value)
        super(ComplexAmount, self).endElement(name, value, connection)