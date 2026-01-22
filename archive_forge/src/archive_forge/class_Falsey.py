from copy import copy
from . import ast
from .visitor_meta import QUERY_DOCUMENT_KEYS, VisitorMeta
class Falsey(object):

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False