import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
@classmethod
def _get_queryset_methods(cls, queryset_class):

    def create_method(name, method):

        @wraps(method)
        def manager_method(self, *args, **kwargs):
            return getattr(self.get_queryset(), name)(*args, **kwargs)
        return manager_method
    new_methods = {}
    for name, method in inspect.getmembers(queryset_class, predicate=inspect.isfunction):
        if hasattr(cls, name):
            continue
        queryset_only = getattr(method, 'queryset_only', None)
        if queryset_only or (queryset_only is None and name.startswith('_')):
            continue
        new_methods[name] = create_method(name, method)
    return new_methods