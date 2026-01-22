from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _create_xml_parser(user_parser):
    if user_parser is not None:
        return user_parser
    try:
        return etree.XMLParser(resolve_entities=False)
    except TypeError:
        return None