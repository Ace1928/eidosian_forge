import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class JsonPath(_JsonLookupBase):

    def __sql__(self, ctx):
        return ctx.sql(self.node).literal('#>' if self._as_json else '#>>').sql(Value('{%s}' % ','.join(map(str, self.parts))))