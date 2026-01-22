from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
class XFloat(XBuiltin):
    """Represents an XSD <xsd:float/> built-in type."""

    @staticmethod
    def translate(value, topython=True):
        if topython:
            if isinstance(value, str) and value:
                return float(value)
        else:
            return value