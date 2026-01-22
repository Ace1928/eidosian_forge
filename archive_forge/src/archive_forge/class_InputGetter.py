import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class InputGetter:
    """
    An accessor that represents all the input fields in a form.

    You can get fields by name from this, with
    ``form.inputs['field_name']``.  If there are a set of checkboxes
    with the same name, they are returned as a list (a `CheckboxGroup`
    which also allows value setting).  Radio inputs are handled
    similarly.  Use ``.keys()`` and ``.items()`` to process all fields
    in this way.

    You can also iterate over this to get all input elements.  This
    won't return the same thing as if you get all the names, as
    checkboxes and radio elements are returned individually.
    """

    def __init__(self, form):
        self.form = form

    def __repr__(self):
        return '<%s for form %s>' % (self.__class__.__name__, self.form._name())

    def __getitem__(self, name):
        fields = [field for field in self if field.name == name]
        if not fields:
            raise KeyError('No input element with the name %r' % name)
        input_type = fields[0].get('type')
        if input_type == 'radio' and len(fields) > 1:
            group = RadioGroup(fields)
            group.name = name
            return group
        elif input_type == 'checkbox' and len(fields) > 1:
            group = CheckboxGroup(fields)
            group.name = name
            return group
        else:
            return fields[0]

    def __contains__(self, name):
        for field in self:
            if field.name == name:
                return True
        return False

    def keys(self):
        """
        Returns all unique field names, in document order.

        :return: A list of all unique field names.
        """
        names = []
        seen = {None}
        for el in self:
            name = el.name
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def items(self):
        """
        Returns all fields with their names, similar to dict.items().

        :return: A list of (name, field) tuples.
        """
        items = []
        seen = set()
        for el in self:
            name = el.name
            if name not in seen:
                seen.add(name)
                items.append((name, self[name]))
        return items

    def __iter__(self):
        return self.form.iter('select', 'input', 'textarea')

    def __len__(self):
        return sum((1 for _ in self))