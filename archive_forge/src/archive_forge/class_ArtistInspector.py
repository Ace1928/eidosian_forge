from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
class ArtistInspector:
    """
    A helper class to inspect an `~matplotlib.artist.Artist` and return
    information about its settable properties and their current values.
    """

    def __init__(self, o):
        """
        Initialize the artist inspector with an `Artist` or an iterable of
        `Artist`\\s.  If an iterable is used, we assume it is a homogeneous
        sequence (all `Artist`\\s are of the same type) and it is your
        responsibility to make sure this is so.
        """
        if not isinstance(o, Artist):
            if np.iterable(o):
                o = list(o)
                if len(o):
                    o = o[0]
        self.oorig = o
        if not isinstance(o, type):
            o = type(o)
        self.o = o
        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        Get a dict mapping property fullnames to sets of aliases for each alias
        in the :class:`~matplotlib.artist.ArtistInspector`.

        e.g., for lines::

          {'markerfacecolor': {'mfc'},
           'linewidth'      : {'lw'},
          }
        """
        names = [name for name in dir(self.o) if name.startswith(('set_', 'get_')) and callable(getattr(self.o, name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func):
                continue
            propname = re.search(f'`({name[:4]}.*)`', inspect.getdoc(func)).group(1)
            aliases.setdefault(propname[4:], set()).add(name[4:])
        return aliases
    _get_valid_values_regex = re.compile('\\n\\s*(?:\\.\\.\\s+)?ACCEPTS:\\s*((?:.|\\n)*?)(?:$|(?:\\n\\n))')

    def get_valid_values(self, attr):
        """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the setter for a line that
        begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
        numpydoc-style documentation for the setter's first argument.
        """
        name = 'set_%s' % attr
        if not hasattr(self.o, name):
            raise AttributeError(f'{self.o} has no function {name}')
        func = getattr(self.o, name)
        docstring = inspect.getdoc(func)
        if docstring is None:
            return 'unknown'
        if docstring.startswith('Alias for '):
            return None
        match = self._get_valid_values_regex.search(docstring)
        if match is not None:
            return re.sub('\n *', ' ', match.group(1))
        param_name = func.__code__.co_varnames[1]
        match = re.search(f'(?m)^ *\\*?{param_name} : (.+)', docstring)
        if match:
            return match.group(1)
        return 'unknown'

    def _replace_path(self, source_class):
        """
        Changes the full path to the public API path that is used
        in sphinx. This is needed for links to work.
        """
        replace_dict = {'_base._AxesBase': 'Axes', '_axes.Axes': 'Axes'}
        for key, value in replace_dict.items():
            source_class = source_class.replace(key, value)
        return source_class

    def get_setters(self):
        """
        Get the attribute strings with setters for object.

        For example, for a line, return ``['markerfacecolor', 'linewidth',
        ....]``.
        """
        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'):
                continue
            func = getattr(self.o, name)
            if not callable(func) or self.number_of_parameters(func) < 2 or self.is_alias(func):
                continue
            setters.append(name[4:])
        return setters

    @staticmethod
    @cache
    def number_of_parameters(func):
        """Return number of parameters of the callable *func*."""
        return len(inspect.signature(func).parameters)

    @staticmethod
    @cache
    def is_alias(method):
        """
        Return whether the object *method* is an alias for another method.
        """
        ds = inspect.getdoc(method)
        if ds is None:
            return False
        return ds.startswith('Alias for ')

    def aliased_name(self, s):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME'.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
        aliases = ''.join((' or %s' % x for x in sorted(self.aliasd.get(s, []))))
        return s + aliases
    _NOT_LINKABLE = {'matplotlib.image._ImageBase.set_alpha', 'matplotlib.image._ImageBase.set_array', 'matplotlib.image._ImageBase.set_data', 'matplotlib.image._ImageBase.set_filternorm', 'matplotlib.image._ImageBase.set_filterrad', 'matplotlib.image._ImageBase.set_interpolation', 'matplotlib.image._ImageBase.set_interpolation_stage', 'matplotlib.image._ImageBase.set_resample', 'matplotlib.text._AnnotationBase.set_annotation_clip'}

    def aliased_name_rest(self, s, target):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME',
        formatted for reST.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
        if target in self._NOT_LINKABLE:
            return f'``{s}``'
        aliases = ''.join((' or %s' % x for x in sorted(self.aliasd.get(s, []))))
        return f':meth:`{s} <{target}>`{aliases}'

    def pprint_setters(self, prop=None, leadingspace=2):
        """
        If *prop* is *None*, return a list of strings of all settable
        properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return f'{pad}{prop}: {accepts}'
        lines = []
        for prop in sorted(self.get_setters()):
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)
            lines.append(f'{pad}{name}: {accepts}')
        return lines

    def pprint_setters_rest(self, prop=None, leadingspace=4):
        """
        If *prop* is *None*, return a list of reST-formatted strings of all
        settable properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of "property : valid"
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return f'{pad}{prop}: {accepts}'
        prop_and_qualnames = []
        for prop in sorted(self.get_setters()):
            for cls in self.o.__mro__:
                method = getattr(cls, f'set_{prop}', None)
                if method and method.__doc__ is not None:
                    break
            else:
                method = getattr(self.o, f'set_{prop}')
            prop_and_qualnames.append((prop, f'{method.__module__}.{method.__qualname__}'))
        names = [self.aliased_name_rest(prop, target).replace('_base._AxesBase', 'Axes').replace('_axes.Axes', 'Axes') for prop, target in prop_and_qualnames]
        accepts = [self.get_valid_values(prop) for prop, _ in prop_and_qualnames]
        col0_len = max((len(n) for n in names))
        col1_len = max((len(a) for a in accepts))
        table_formatstr = pad + '   ' + '=' * col0_len + '   ' + '=' * col1_len
        return ['', pad + '.. table::', pad + '   :class: property-table', '', table_formatstr, pad + '   ' + 'Property'.ljust(col0_len) + '   ' + 'Description'.ljust(col1_len), table_formatstr, *[pad + '   ' + n.ljust(col0_len) + '   ' + a.ljust(col1_len) for n, a in zip(names, accepts)], table_formatstr, '']

    def properties(self):
        """Return a dictionary mapping property name -> value."""
        o = self.oorig
        getters = [name for name in dir(o) if name.startswith('get_') and callable(getattr(o, name))]
        getters.sort()
        d = {}
        for name in getters:
            func = getattr(o, name)
            if self.is_alias(func):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    val = func()
            except Exception:
                continue
            else:
                d[name[4:]] = val
        return d

    def pprint_getters(self):
        """Return the getters and actual values as list of strings."""
        lines = []
        for name, val in sorted(self.properties().items()):
            if getattr(val, 'shape', ()) != () and len(val) > 6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)
            s = s.replace('\n', ' ')
            if len(s) > 50:
                s = s[:50] + '...'
            name = self.aliased_name(name)
            lines.append(f'    {name} = {s}')
        return lines