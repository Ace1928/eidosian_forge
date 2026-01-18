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
@staticmethod
@filter_method
def type_qname(obj, suffix=None, unnamed='none', sep='__'):
    """
        Get the unqualified name of the XSD type. Invalid
        chars for identifiers are replaced by an underscore.

        :param obj: an instance of (XsdType|XsdAttribute|XsdElement).
        :param suffix: force a suffix. For default removes '_type' or 'Type' suffixes.
        :param unnamed: value for unnamed XSD types. Defaults to 'none'.
        :param sep: the replacement for colon. Defaults to double underscore.
        :return: str
        """
    if isinstance(obj, XsdType):
        qname = obj.prefixed_name or unnamed
    elif isinstance(obj, (XsdElement, XsdAttribute)):
        qname = obj.type.prefixed_name or unnamed
    else:
        qname = unnamed
    if qname.endswith('Type'):
        qname = qname[:-4]
    elif qname.endswith('_type'):
        qname = qname[:-5]
    if suffix:
        qname = '{}{}'.format(qname, suffix)
    return qname.replace('.', '_').replace('-', '_').replace(':', sep)