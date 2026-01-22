import sys
import os
from numpy.core import dtype
from numpy.core import numerictypes as _numerictypes
from numpy.core.function_base import add_newdoc
def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    o = getattr(_numerictypes, obj)
    character_code = dtype(o).char
    canonical_name_doc = '' if obj == o.__name__ else f':Canonical name: `numpy.{obj}`\n    '
    if fixed_aliases:
        alias_doc = ''.join((f':Alias: `numpy.{alias}`\n    ' for alias in fixed_aliases))
    else:
        alias_doc = ''
    alias_doc += ''.join((f'{_doc_alias_string} `numpy.{alias}`: {doc}.\n    ' for alias_type, alias, doc in possible_aliases if alias_type is o))
    docstring = f"\n    {doc.strip()}\n\n    :Character code: ``'{character_code}'``\n    {canonical_name_doc}{alias_doc}\n    "
    add_newdoc('numpy.core.numerictypes', obj, docstring)