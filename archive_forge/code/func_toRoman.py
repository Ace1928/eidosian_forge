import re
def toRoman(n):
    """convert integer to Roman numeral"""
    if not 0 < n < 5000:
        raise OutOfRangeError('number out of range (must be 1..4999)')
    if int(n) != n:
        raise NotIntegerError('decimals can not be converted')
    result = ''
    for numeral, integer in romanNumeralMap:
        while n >= integer:
            result += numeral
            n -= integer
    return result