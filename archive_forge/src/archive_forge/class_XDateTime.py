from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XDateTime(XBuiltin):
    """Represents an XSD <xsd:datetime/> built-in type."""

    @staticmethod
    def translate(value, topython=True):
        if topython:
            if isinstance(value, str) and value:
                return DateTime(value).value
        else:
            if isinstance(value, datetime.datetime):
                return DateTime(value)
            return value