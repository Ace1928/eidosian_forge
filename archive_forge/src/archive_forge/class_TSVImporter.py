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
class TSVImporter(CSVImporter):

    def load(self, file_obj, header=True, **kwargs):
        kwargs.setdefault('delimiter', '\t')
        return super(TSVImporter, self).load(file_obj, header, **kwargs)