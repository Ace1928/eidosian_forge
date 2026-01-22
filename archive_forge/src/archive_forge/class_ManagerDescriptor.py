import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
class ManagerDescriptor:

    def __init__(self, manager):
        self.manager = manager

    def __get__(self, instance, cls=None):
        if instance is not None:
            raise AttributeError("Manager isn't accessible via %s instances" % cls.__name__)
        if cls._meta.abstract:
            raise AttributeError("Manager isn't available; %s is abstract" % (cls._meta.object_name,))
        if cls._meta.swapped:
            raise AttributeError("Manager isn't available; '%s' has been swapped for '%s'" % (cls._meta.label, cls._meta.swapped))
        return cls._meta.managers_map[self.manager.name]