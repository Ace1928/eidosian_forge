import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class InputElement(InputMixin, HtmlElement):
    """
    Represents an ``<input>`` element.

    You can get the type with ``.type`` (which is lower-cased and
    defaults to ``'text'``).

    Also you can get and set the value with ``.value``

    Checkboxes and radios have the attribute ``input.checkable ==
    True`` (for all others it is false) and a boolean attribute
    ``.checked``.

    """

    @property
    def value(self):
        """
        Get/set the value of this element, using the ``value`` attribute.

        Also, if this is a checkbox and it has no value, this defaults
        to ``'on'``.  If it is a checkbox or radio that is not
        checked, this returns None.
        """
        if self.checkable:
            if self.checked:
                return self.get('value') or 'on'
            else:
                return None
        return self.get('value')

    @value.setter
    def value(self, value):
        if self.checkable:
            if not value:
                self.checked = False
            else:
                self.checked = True
                if isinstance(value, str):
                    self.set('value', value)
        else:
            self.set('value', value)

    @value.deleter
    def value(self):
        if self.checkable:
            self.checked = False
        elif 'value' in self.attrib:
            del self.attrib['value']

    @property
    def type(self):
        """
        Return the type of this element (using the type attribute).
        """
        return self.get('type', 'text').lower()

    @type.setter
    def type(self, value):
        self.set('type', value)

    @property
    def checkable(self):
        """
        Boolean: can this element be checked?
        """
        return self.type in ('checkbox', 'radio')

    @property
    def checked(self):
        """
        Boolean attribute to get/set the presence of the ``checked``
        attribute.

        You can only use this on checkable input types.
        """
        if not self.checkable:
            raise AttributeError('Not a checkable input type')
        return 'checked' in self.attrib

    @checked.setter
    def checked(self, value):
        if not self.checkable:
            raise AttributeError('Not a checkable input type')
        if value:
            self.set('checked', '')
        else:
            attrib = self.attrib
            if 'checked' in attrib:
                del attrib['checked']