import copy
import json
import jsonpatch
import warlock.model as warlock
class SchemaProperty(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self.description = kwargs.get('description')
        self.is_base = kwargs.get('is_base', True)