import math
import sys
from flask import abort
from flask import render_template
from flask import request
from peewee import Database
from peewee import DoesNotExist
from peewee import Model
from peewee import Proxy
from peewee import SelectQuery
from playhouse.db_url import connect as db_url_connect
@property
def Model(self):
    if self._app is None:
        database = getattr(self, 'database', None)
        if database is None:
            self.database = Proxy()
    if not hasattr(self, '_model_class'):
        self._model_class = self.get_model_class()
    return self._model_class