import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
class Common(object):
    """Common information to several classes.

    Should not be directly used, several classes are derived from
    this one.
    """

    def __getstate__(self):
        dict = copy.copy(self.obj_dict)
        return dict

    def __setstate__(self, state):
        self.obj_dict = state

    def __get_attribute__(self, attr):
        """Look for default attributes for this node"""
        attr_val = self.obj_dict['attributes'].get(attr, None)
        if attr_val is None:
            default_node_name = self.obj_dict['type']
            if default_node_name in ('subgraph', 'digraph', 'cluster'):
                default_node_name = 'graph'
            g = self.get_parent_graph()
            if g is not None:
                defaults = g.get_node(default_node_name)
            else:
                return None
            if not isinstance(defaults, (list, tuple)):
                defaults = [defaults]
            for default in defaults:
                attr_val = default.obj_dict['attributes'].get(attr, None)
                if attr_val:
                    return attr_val
        else:
            return attr_val
        return None

    def set_parent_graph(self, parent_graph):
        self.obj_dict['parent_graph'] = parent_graph

    def get_parent_graph(self):
        return self.obj_dict.get('parent_graph', None)

    def set(self, name, value):
        """Set an attribute value by name.

        Given an attribute 'name' it will set its value to 'value'.
        There's always the possibility of using the methods:

            set_'name'(value)

        which are defined for all the existing attributes.
        """
        self.obj_dict['attributes'][name] = value

    def get(self, name):
        """Get an attribute value by name.

        Given an attribute 'name' it will get its value.
        There's always the possibility of using the methods:

            get_'name'()

        which are defined for all the existing attributes.
        """
        return self.obj_dict['attributes'].get(name, None)

    def get_attributes(self):
        """Get attributes of the object"""
        return self.obj_dict['attributes']

    def set_sequence(self, seq):
        """Set sequence"""
        self.obj_dict['sequence'] = seq

    def get_sequence(self):
        """Get sequence"""
        return self.obj_dict['sequence']

    def create_attribute_methods(self, obj_attributes):
        for attr in obj_attributes:
            self.__setattr__('set_' + attr, lambda x, a=attr: self.obj_dict['attributes'].__setitem__(a, x))
            self.__setattr__('get_' + attr, lambda a=attr: self.__get_attribute__(a))