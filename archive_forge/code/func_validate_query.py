import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
@staticmethod
def validate_query(query):
    """
        Simple helper function to indicate whether a search query is a
        valid FTS5 query. Note: this simply looks at the characters being
        used, and is not guaranteed to catch all problematic queries.
        """
    tokens = _quote_re.findall(query)
    for token in tokens:
        if token.startswith('"') and token.endswith('"'):
            continue
        if set(token) & _invalid_ascii:
            return False
    return True