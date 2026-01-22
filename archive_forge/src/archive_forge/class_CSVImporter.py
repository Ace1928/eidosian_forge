import csv
import datetime
from decimal import Decimal
import json
import operator
import sys
import uuid
from peewee import *
from playhouse.db_url import connect
from playhouse.migrate import migrate
from playhouse.migrate import SchemaMigrator
from playhouse.reflection import Introspector
class CSVImporter(Importer):

    def load(self, file_obj, header=True, **kwargs):
        count = 0
        reader = csv.reader(file_obj, **kwargs)
        if header:
            try:
                header_keys = next(reader)
            except StopIteration:
                return count
            if self.strict:
                header_fields = []
                for idx, key in enumerate(header_keys):
                    if key in self.columns:
                        header_fields.append((idx, self.columns[key]))
            else:
                header_fields = list(enumerate(header_keys))
        else:
            header_fields = list(enumerate(self.model._meta.sorted_fields))
        if not header_fields:
            return count
        for row in reader:
            obj = {}
            for idx, field in header_fields:
                if self.strict:
                    obj[field.name] = field.python_value(row[idx])
                else:
                    obj[field] = row[idx]
            self.table.insert(**obj)
            count += 1
        return count