import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class FormElement(HtmlElement):
    """
    Represents a <form> element.
    """

    @property
    def inputs(self):
        """
        Returns an accessor for all the input elements in the form.

        See `InputGetter` for more information about the object.
        """
        return InputGetter(self)

    @property
    def fields(self):
        """
        Dictionary-like object that represents all the fields in this
        form.  You can set values in this dictionary to effect the
        form.
        """
        return FieldsDict(self.inputs)

    @fields.setter
    def fields(self, value):
        fields = self.fields
        prev_keys = fields.keys()
        for key, value in value.items():
            if key in prev_keys:
                prev_keys.remove(key)
            fields[key] = value
        for key in prev_keys:
            if key is None:
                continue
            fields[key] = None

    def _name(self):
        if self.get('name'):
            return self.get('name')
        elif self.get('id'):
            return '#' + self.get('id')
        iter_tags = self.body.iter
        forms = list(iter_tags('form'))
        if not forms:
            forms = list(iter_tags('{%s}form' % XHTML_NAMESPACE))
        return str(forms.index(self))

    def form_values(self):
        """
        Return a list of tuples of the field values for the form.
        This is suitable to be passed to ``urllib.urlencode()``.
        """
        results = []
        for el in self.inputs:
            name = el.name
            if not name or 'disabled' in el.attrib:
                continue
            tag = _nons(el.tag)
            if tag == 'textarea':
                results.append((name, el.value))
            elif tag == 'select':
                value = el.value
                if el.multiple:
                    for v in value:
                        results.append((name, v))
                elif value is not None:
                    results.append((name, el.value))
            else:
                assert tag == 'input', 'Unexpected tag: %r' % el
                if el.checkable and (not el.checked):
                    continue
                if el.type in ('submit', 'image', 'reset', 'file'):
                    continue
                value = el.value
                if value is not None:
                    results.append((name, el.value))
        return results

    @property
    def action(self):
        """
        Get/set the form's ``action`` attribute.
        """
        base_url = self.base_url
        action = self.get('action')
        if base_url and action is not None:
            return urljoin(base_url, action)
        else:
            return action

    @action.setter
    def action(self, value):
        self.set('action', value)

    @action.deleter
    def action(self):
        attrib = self.attrib
        if 'action' in attrib:
            del attrib['action']

    @property
    def method(self):
        """
        Get/set the form's method.  Always returns a capitalized
        string, and defaults to ``'GET'``
        """
        return self.get('method', 'GET').upper()

    @method.setter
    def method(self, value):
        self.set('method', value.upper())