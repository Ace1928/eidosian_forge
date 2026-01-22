import os
import re
import sys
import inspect
import logging
from abc import ABC, ABCMeta
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, \
from elementpath import datatypes
import xmlschema
from xmlschema.validators import XsdType, XsdElement, XsdAttribute
from xmlschema.names import XSD_NAMESPACE
class PythonGenerator(AbstractGenerator):
    """A Python code generator for XSD schemas."""
    formal_language = 'Python'
    searchpaths = ['templates/python/']
    builtin_types = {'string': 'str', 'decimal': 'decimal.Decimal', 'float': 'float', 'double': 'float', 'date': 'datatypes.Date10', 'dateTime': 'datatypes.DateTime10', 'gDay': 'datatypes.GregorianDay', 'gMonth': 'datatypes.GregorianMonth', 'gMonthDay': 'datatypes.GregorianMonthDay', 'gYear': 'datatypes.GregorianYear10', 'gYearMonth': 'datatypes.GregorianYearMonth10', 'time': 'datatypes.Time', 'duration': 'datatypes.Duration', 'QName': 'datatypes.QName', 'NOTATION': 'datatypes.DateTime10', 'anyURI': 'datatypes.AnyURI', 'boolean': 'bool', 'base64Binary': 'datatypes.Base64Binary', 'hexBinary': 'datatypes.HexBinary', 'normalizedString': 'str', 'token': 'str', 'language': 'str', 'Name': 'str', 'NCName': 'str', 'ID': 'str', 'IDREF': 'str', 'ENTITY': 'str', 'NMTOKEN': 'str', 'integer': 'int', 'long': 'int', 'int': 'int', 'short': 'int', 'byte': 'int', 'nonNegativeInteger': 'int', 'positiveInteger': 'int', 'unsignedLong': 'int', 'unsignedInt': 'int', 'unsignedShort': 'int', 'unsignedByte': 'int', 'nonPositiveInteger': 'int', 'negativeInteger': 'int', 'dateTimeStamp': 'datatypes.DateTimeStamp10', 'dayTimeDuration': 'datatypes.DayTimeDuration', 'yearMonthDuration': 'datatypes.YearMonthDuration'}