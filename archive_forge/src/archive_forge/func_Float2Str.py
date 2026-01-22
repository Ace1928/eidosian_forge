from decimal import Decimal
from MySQLdb._mysql import string_literal
from MySQLdb.constants import FIELD_TYPE, FLAG
from MySQLdb.times import (
from MySQLdb._exceptions import ProgrammingError
import array
def Float2Str(o, d):
    s = repr(o)
    if s in ('inf', '-inf', 'nan'):
        raise ProgrammingError('%s can not be used with MySQL' % s)
    if 'e' not in s:
        s += 'e0'
    return s