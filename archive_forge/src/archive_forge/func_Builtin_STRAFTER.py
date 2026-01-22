from __future__ import annotations
import datetime as py_datetime  # naming conflict with function within this module
import hashlib
import math
import operator as pyop  # python operators
import random
import re
import uuid
import warnings
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from functools import reduce
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union, overload
from urllib.parse import quote
import isodate
from pyparsing import ParseResults
from rdflib.namespace import RDF, XSD
from rdflib.plugins.sparql.datatypes import (
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import (
from rdflib.term import (
def Builtin_STRAFTER(expr: Expr, ctx) -> Literal:
    """
    http://www.w3.org/TR/sparql11-query/#func-strafter
    """
    a = expr.arg1
    b = expr.arg2
    _compatibleStrings(a, b)
    i = a.find(b)
    if i == -1:
        return Literal('')
    else:
        return Literal(a[i + len(b):], lang=a.language, datatype=a.datatype)