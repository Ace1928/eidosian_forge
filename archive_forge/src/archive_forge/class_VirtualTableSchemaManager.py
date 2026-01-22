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
class VirtualTableSchemaManager(SchemaManager):

    def _create_virtual_table(self, safe=True, **options):
        options = self.model.clean_options(merge_dict(self.model._meta.options, options))
        ctx = self._create_context()
        ctx.literal('CREATE VIRTUAL TABLE ')
        if safe:
            ctx.literal('IF NOT EXISTS ')
        ctx.sql(self.model).literal(' USING ')
        ext_module = self.model._meta.extension_module
        if isinstance(ext_module, Node):
            return ctx.sql(ext_module)
        ctx.sql(SQL(ext_module)).literal(' ')
        arguments = []
        meta = self.model._meta
        if meta.prefix_arguments:
            arguments.extend([SQL(a) for a in meta.prefix_arguments])
        for field in meta.sorted_fields:
            if isinstance(field, RowIDField) or field._hidden:
                continue
            field_def = [Entity(field.column_name)]
            if field.unindexed:
                field_def.append(SQL('UNINDEXED'))
            arguments.append(NodeList(field_def))
        if meta.arguments:
            arguments.extend([SQL(a) for a in meta.arguments])
        if options:
            arguments.extend(self._create_table_option_sql(options))
        return ctx.sql(EnclosedNodeList(arguments))

    def _create_table(self, safe=True, **options):
        if issubclass(self.model, VirtualModel):
            return self._create_virtual_table(safe, **options)
        return super(VirtualTableSchemaManager, self)._create_table(safe, **options)