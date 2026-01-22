from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XDecimal(XBuiltin):
    """
    Represents an XSD <xsd:decimal/> built-in type.

    Excerpt from the XSD datatype specification
    (http://www.w3.org/TR/2004/REC-xmlschema-2-20041028):

    > 3.2.3 decimal
    >
    > [Definition:] decimal represents a subset of the real numbers, which can
    > be represented by decimal numerals. The ·value space· of decimal is the
    > set of numbers that can be obtained by multiplying an integer by a
    > non-positive power of ten, i.e., expressible as i × 10^-n where i and n
    > are integers and n >= 0. Precision is not reflected in this value space;
    > the number 2.0 is not distinct from the number 2.00. The ·order-relation·
    > on decimal is the order relation on real numbers, restricted to this
    > subset.
    >
    > 3.2.3.1 Lexical representation
    >
    > decimal has a lexical representation consisting of a finite-length
    > sequence of decimal digits (#x30-#x39) separated by a period as a decimal
    > indicator. An optional leading sign is allowed. If the sign is omitted,
    > "+" is assumed. Leading and trailing zeroes are optional. If the
    > fractional part is zero, the period and following zero(es) can be
    > omitted. For example: -1.23, 12678967.543233, +100000.00, 210.

    """
    if sys.version_info < (2, 7):
        _decimal_canonical = staticmethod(lambda decimal: decimal)
    else:
        _decimal_canonical = decimal.Decimal.canonical

    @staticmethod
    def _decimal_to_xsd_format(value):
        """
        Converts a decimal.Decimal value to its XSD decimal type value.

        Result is a string containing the XSD decimal type's lexical value
        representation. The conversion is done without any precision loss.

        Note that Python's native decimal.Decimal string representation will
        not do here as the lexical representation desired here does not allow
        representing decimal values using float-like `<mantissa>E<exponent>'
        format, e.g. 12E+30 or 0.10006E-12.

        """
        value = XDecimal._decimal_canonical(value)
        negative, digits, exponent = value.as_tuple()
        assert digits
        assert digits[0] != 0 or len(digits) == 1
        result = []
        if negative:
            result.append('-')
        if exponent >= 0:
            result.extend((str(x) for x in digits))
            result.extend('0' * exponent)
            return ''.join(result)
        digit_count = len(digits)
        point_offset = digit_count + exponent
        fractional_digit_count = min(digit_count, -exponent)
        while fractional_digit_count and digits[digit_count - 1] == 0:
            digit_count -= 1
            fractional_digit_count -= 1
        if point_offset <= 0:
            result.append('0')
            if digit_count > 0:
                result.append('.')
                result.append('0' * -point_offset)
                result.extend((str(x) for x in digits[:digit_count]))
        else:
            result.extend((str(x) for x in digits[:point_offset]))
            if point_offset < digit_count:
                result.append('.')
                result.extend((str(x) for x in digits[point_offset:digit_count]))
        return ''.join(result)

    @classmethod
    def translate(cls, value, topython=True):
        if topython:
            if isinstance(value, str) and value:
                return decimal.Decimal(value)
        else:
            if isinstance(value, decimal.Decimal):
                return cls._decimal_to_xsd_format(value)
            return value