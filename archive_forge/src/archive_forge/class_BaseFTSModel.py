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
class BaseFTSModel(VirtualModel):

    @classmethod
    def clean_options(cls, options):
        content = options.get('content')
        prefix = options.get('prefix')
        tokenize = options.get('tokenize')
        if isinstance(content, basestring) and content == '':
            options['content'] = "''"
        elif isinstance(content, Field):
            options['content'] = Entity(content.model._meta.table_name, content.column_name)
        if prefix:
            if isinstance(prefix, (list, tuple)):
                prefix = ','.join([str(i) for i in prefix])
            options['prefix'] = "'%s'" % prefix.strip("' ")
        if tokenize and cls._meta.extension_module.lower() == 'fts5':
            options['tokenize'] = '"%s"' % tokenize
        return options