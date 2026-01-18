import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
def wrapped_iterator():
    for row in query.dicts().iterator():
        identifier = row.pop(key)
        model = mapping[identifier]
        yield model(**row)