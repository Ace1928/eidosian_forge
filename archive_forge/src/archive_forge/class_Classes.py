import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class Classes(MutableSet):
    """Provides access to an element's class attribute as a set-like collection.
    Usage::

        >>> el = fromstring('<p class="hidden large">Text</p>')
        >>> classes = el.classes  # or: classes = Classes(el.attrib)
        >>> classes |= ['block', 'paragraph']
        >>> el.get('class')
        'hidden large block paragraph'
        >>> classes.toggle('hidden')
        False
        >>> el.get('class')
        'large block paragraph'
        >>> classes -= ('some', 'classes', 'block')
        >>> el.get('class')
        'large paragraph'
    """

    def __init__(self, attributes):
        self._attributes = attributes
        self._get_class_value = partial(attributes.get, 'class', '')

    def add(self, value):
        """
        Add a class.

        This has no effect if the class is already present.
        """
        if not value or re.search('\\s', value):
            raise ValueError('Invalid class name: %r' % value)
        classes = self._get_class_value().split()
        if value in classes:
            return
        classes.append(value)
        self._attributes['class'] = ' '.join(classes)

    def discard(self, value):
        """
        Remove a class if it is currently present.

        If the class is not present, do nothing.
        """
        if not value or re.search('\\s', value):
            raise ValueError('Invalid class name: %r' % value)
        classes = [name for name in self._get_class_value().split() if name != value]
        if classes:
            self._attributes['class'] = ' '.join(classes)
        elif 'class' in self._attributes:
            del self._attributes['class']

    def remove(self, value):
        """
        Remove a class; it must currently be present.

        If the class is not present, raise a KeyError.
        """
        if not value or re.search('\\s', value):
            raise ValueError('Invalid class name: %r' % value)
        super().remove(value)

    def __contains__(self, name):
        classes = self._get_class_value()
        return name in classes and name in classes.split()

    def __iter__(self):
        return iter(self._get_class_value().split())

    def __len__(self):
        return len(self._get_class_value().split())

    def update(self, values):
        """
        Add all names from 'values'.
        """
        classes = self._get_class_value().split()
        extended = False
        for value in values:
            if value not in classes:
                classes.append(value)
                extended = True
        if extended:
            self._attributes['class'] = ' '.join(classes)

    def toggle(self, value):
        """
        Add a class name if it isn't there yet, or remove it if it exists.

        Returns true if the class was added (and is now enabled) and
        false if it was removed (and is now disabled).
        """
        if not value or re.search('\\s', value):
            raise ValueError('Invalid class name: %r' % value)
        classes = self._get_class_value().split()
        try:
            classes.remove(value)
            enabled = False
        except ValueError:
            classes.append(value)
            enabled = True
        if classes:
            self._attributes['class'] = ' '.join(classes)
        else:
            del self._attributes['class']
        return enabled