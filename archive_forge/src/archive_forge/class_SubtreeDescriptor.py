from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
class SubtreeDescriptor(Descriptor):
    """Implements the descriptor interface for nested configuration
     objects."""

    def __init__(self, subtree_class):
        super(SubtreeDescriptor, self).__init__()
        self.name = '<unnamed>'
        self.subtree_class = subtree_class

    def get(self, obj):
        if not hasattr(obj, '_' + self.name):
            setattr(obj, '_' + self.name, self.subtree_class())
        return getattr(obj, '_' + self.name)

    def __get__(self, obj, _objtype):
        return self.get(obj)

    def __set__(self, obj, value):
        raise NotImplementedError('Assignment to subtree objects is not supported')

    def __set_name__(self, owner, name):
        owner._field_registry.append(self)
        if sys.version_info < (3, 0, 0):
            name = name.decode('utf-8')
        self.name = name

    def consume_value(self, obj, value_dict):
        """Convenience method to consume keyword arguments directly from the
    descriptor interface."""
        if not isinstance(value_dict, dict):
            raise ValueError('value_dict for {}.{} must be a dictionary, not a {}'.format(type(obj).__name__, self.name, type(value_dict).__name__))
        self.get(obj).consume_known(value_dict)
        warn_unused(value_dict)

    def legacy_shim_consume(self, obj, kwargs):
        """Consume config variable assignments from the root of the config
    tree. This is the legacy config style and will likely be deprecated
    soon."""
        self.get(obj).legacy_consume(kwargs)

    def add_to_argparser(self, argparser):
        self.subtree_class.add_to_argparser(argparser)