from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XTime(XBuiltin):
    """Represents an XSD <xsd:time/> built-in type."""

    @staticmethod
    def translate(value, topython=True):
        if topython:
            if isinstance(value, str) and value:
                return Time(value).value
        else:
            if isinstance(value, datetime.time):
                return Time(value)
            return value