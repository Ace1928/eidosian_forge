import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class InputMixin:
    """
    Mix-in for all input elements (input, select, and textarea)
    """

    @property
    def name(self):
        """
        Get/set the name of the element
        """
        return self.get('name')

    @name.setter
    def name(self, value):
        self.set('name', value)

    @name.deleter
    def name(self):
        attrib = self.attrib
        if 'name' in attrib:
            del attrib['name']

    def __repr__(self):
        type_name = getattr(self, 'type', None)
        if type_name:
            type_name = ' type=%r' % type_name
        else:
            type_name = ''
        return '<%s %x name=%r%s>' % (self.__class__.__name__, id(self), self.name, type_name)