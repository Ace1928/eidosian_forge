from oslo_utils import timeutils
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.orm import object_mapper
from oslo_db.sqlalchemy import types
class ModelIterator(object):

    def __init__(self, model, columns):
        self.model = model
        self.i = columns

    def __iter__(self):
        return self

    def __next__(self):
        n = next(self.i)
        return (n, getattr(self.model, n))