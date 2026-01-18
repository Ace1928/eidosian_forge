from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys

        Create an object based on the root tag name.

        @param schema: A schema object.
        @type schema: L{schema.Schema}
        @param name: The name.
        @type name: str
        @return: The created object.
        @rtype: L{XBuiltin}

        