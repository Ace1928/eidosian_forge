import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
class ComponentMeta(abc.ABCMeta):

    def __new__(mcs, name, bases, attributes):
        component = abc.ABCMeta.__new__(mcs, name, bases, attributes)
        module = attributes['__module__'].split('.')[0]
        if name == 'Component' or module == 'builtins':
            return component
        _namespace = attributes.get('_namespace', module)
        ComponentRegistry.namespace_to_package[_namespace] = module
        ComponentRegistry.registry.add(module)
        ComponentRegistry.children_props[_namespace][name] = attributes.get('_children_props')
        return component