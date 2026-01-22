import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class SelectElement(InputMixin, HtmlElement):
    """
    ``<select>`` element.  You can get the name with ``.name``.

    ``.value`` will be the value of the selected option, unless this
    is a multi-select element (``<select multiple>``), in which case
    it will be a set-like object.  In either case ``.value_options``
    gives the possible values.

    The boolean attribute ``.multiple`` shows if this is a
    multi-select.
    """

    @property
    def value(self):
        """
        Get/set the value of this select (the selected option).

        If this is a multi-select, this is a set-like object that
        represents all the selected options.
        """
        if self.multiple:
            return MultipleSelectOptions(self)
        options = _options_xpath(self)
        try:
            selected_option = next((el for el in reversed(options) if el.get('selected') is not None))
        except StopIteration:
            try:
                selected_option = next((el for el in options if el.get('disabled') is None))
            except StopIteration:
                return None
        value = selected_option.get('value')
        if value is None:
            value = (selected_option.text or '').strip()
        return value

    @value.setter
    def value(self, value):
        if self.multiple:
            if isinstance(value, str):
                raise TypeError('You must pass in a sequence')
            values = self.value
            values.clear()
            values.update(value)
            return
        checked_option = None
        if value is not None:
            for el in _options_xpath(self):
                opt_value = el.get('value')
                if opt_value is None:
                    opt_value = (el.text or '').strip()
                if opt_value == value:
                    checked_option = el
                    break
            else:
                raise ValueError('There is no option with the value of %r' % value)
        for el in _options_xpath(self):
            if 'selected' in el.attrib:
                del el.attrib['selected']
        if checked_option is not None:
            checked_option.set('selected', '')

    @value.deleter
    def value(self):
        if self.multiple:
            self.value.clear()
        else:
            self.value = None

    @property
    def value_options(self):
        """
        All the possible values this select can have (the ``value``
        attribute of all the ``<option>`` elements.
        """
        options = []
        for el in _options_xpath(self):
            value = el.get('value')
            if value is None:
                value = (el.text or '').strip()
            options.append(value)
        return options

    @property
    def multiple(self):
        """
        Boolean attribute: is there a ``multiple`` attribute on this element.
        """
        return 'multiple' in self.attrib

    @multiple.setter
    def multiple(self, value):
        if value:
            self.set('multiple', '')
        elif 'multiple' in self.attrib:
            del self.attrib['multiple']