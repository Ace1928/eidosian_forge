import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
class sanitize_identifier_fn(param.ParameterizedFunction):
    """
    Sanitizes group/label values for use in AttrTree attribute
    access.

    Special characters are sanitized using their (lowercase) unicode
    name using the unicodedata module. For instance:

    >>> unicodedata.name(u'$').lower()
    'dollar sign'

    As these names are often very long, this parameterized function
    allows filtered, substitutions and transforms to help shorten these
    names appropriately.
    """
    capitalize = param.Boolean(default=True, doc="\n       Whether the first letter should be converted to\n       uppercase. Note, this will only be applied to ASCII characters\n       in order to make sure paths aren't confused with method\n       names.")
    eliminations = param.List(default=['extended', 'accent', 'small', 'letter', 'sign', 'digit', 'latin', 'greek', 'arabic-indic', 'with', 'dollar'], doc='\n       Lowercase strings to be eliminated from the unicode names in\n       order to shorten the sanitized name ( lowercase). Redundant\n       strings should be removed but too much elimination could cause\n       two unique strings to map to the same sanitized output.')
    substitutions = param.Dict(default={'circumflex': 'power', 'asterisk': 'times', 'solidus': 'over'}, doc="\n       Lowercase substitutions of substrings in unicode names. For\n       instance the ^ character has the name 'circumflex accent' even\n       though it is more typically used for exponentiation. Note that\n       substitutions occur after filtering and that there should be no\n       ordering dependence between substitutions.")
    transforms = param.List(default=[capitalize_unicode_name], doc='\n       List of string transformation functions to apply after\n       filtering and substitution in order to further compress the\n       unicode name. For instance, the default capitalize_unicode_name\n       function will turn the string "capital delta" into "Delta".')
    disallowed = param.List(default=['trait_names', '_ipython_display_', '_getAttributeNames'], doc='\n       An explicit list of name that should not be allowed as\n       attribute names on Tree objects.\n\n       By default, prevents IPython from creating an entry called\n       Trait_names due to an inconvenient getattr check (during\n       tab-completion).')
    disable_leading_underscore = param.Boolean(default=False, doc='\n       Whether leading underscores should be allowed to be sanitized\n       with the leading prefix.')
    aliases = param.Dict(default={}, doc='\n       A dictionary of aliases mapping long strings to their short,\n       sanitized equivalents')
    prefix = 'A_'
    _lookup_table = param.Dict(default={}, doc='\n       Cache of previously computed sanitizations')

    @param.parameterized.bothmethod
    def add_aliases(self_or_cls, **kwargs):
        """
        Conveniently add new aliases as keyword arguments. For instance
        you can add a new alias with add_aliases(short='Longer string')
        """
        self_or_cls.aliases.update({v: k for k, v in kwargs.items()})

    @param.parameterized.bothmethod
    def remove_aliases(self_or_cls, aliases):
        """
        Remove a list of aliases.
        """
        for k, v in self_or_cls.aliases.items():
            if v in aliases:
                self_or_cls.aliases.pop(k)

    @param.parameterized.bothmethod
    def allowable(self_or_cls, name, disable_leading_underscore=None):
        disabled_reprs = ['javascript', 'jpeg', 'json', 'latex', 'latex', 'pdf', 'png', 'svg', 'markdown']
        disabled_ = self_or_cls.disable_leading_underscore if disable_leading_underscore is None else disable_leading_underscore
        if disabled_ and name.startswith('_'):
            return False
        isrepr = any((f'_repr_{el}_' == name for el in disabled_reprs))
        return name not in self_or_cls.disallowed and (not isrepr)

    @param.parameterized.bothmethod
    def prefixed(self, identifier):
        """
        Whether or not the identifier will be prefixed.
        Strings that require the prefix are generally not recommended.
        """
        invalid_starting = ['Mn', 'Mc', 'Nd', 'Pc']
        if identifier.startswith('_'):
            return True
        return unicodedata.category(identifier[0]) in invalid_starting

    @param.parameterized.bothmethod
    def remove_diacritics(self_or_cls, identifier):
        """
        Remove diacritics and accents from the input leaving other
        unicode characters alone."""
        chars = ''
        for c in identifier:
            replacement = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
            if replacement != '':
                chars += bytes_to_unicode(replacement)
            else:
                chars += c
        return chars

    @param.parameterized.bothmethod
    def shortened_character_name(self_or_cls, c, eliminations=None, substitutions=None, transforms=None):
        """
        Given a unicode character c, return the shortened unicode name
        (as a list of tokens) by applying the eliminations,
        substitutions and transforms.
        """
        if transforms is None:
            transforms = []
        if substitutions is None:
            substitutions = {}
        if eliminations is None:
            eliminations = []
        name = unicodedata.name(c).lower()
        for elim in eliminations:
            name = name.replace(elim, '')
        for i, o in substitutions.items():
            name = name.replace(i, o)
        for transform in transforms:
            name = transform(name)
        return ' '.join(name.strip().split()).replace(' ', '_').replace('-', '_')

    def __call__(self, name, escape=True):
        if name in [None, '']:
            return name
        elif name in self.aliases:
            return self.aliases[name]
        elif name in self._lookup_table:
            return self._lookup_table[name]
        name = bytes_to_unicode(name)
        if not self.allowable(name):
            raise AttributeError(f'String {name!r} is in the disallowed list of attribute names: {self.disallowed!r}')
        if self.capitalize and name and (name[0] in string.ascii_lowercase):
            name = name[0].upper() + name[1:]
        sanitized = self.sanitize_py3(name)
        if self.prefixed(name):
            sanitized = self.prefix + sanitized
        self._lookup_table[name] = sanitized
        return sanitized

    def _process_underscores(self, tokens):
        """Strip underscores to make sure the number is correct after join"""
        groups = [[str(''.join(el))] if b else list(el) for b, el in itertools.groupby(tokens, lambda k: k == '_')]
        flattened = [el for group in groups for el in group]
        processed = []
        for token in flattened:
            if token == '_':
                continue
            if token.startswith('_'):
                token = str(token[1:])
            if token.endswith('_'):
                token = str(token[:-1])
            processed.append(token)
        return processed

    def sanitize_py3(self, name):
        if not name.isidentifier():
            return '_'.join(self.sanitize(name, lambda c: ('_' + c).isidentifier()))
        else:
            return name

    def sanitize(self, name, valid_fn):
        """Accumulate blocks of hex and separate blocks by underscores"""
        invalid = {'\x07': 'a', '\x08': 'b', '\x0b': 'v', '\x0c': 'f', '\r': 'r'}
        for cc in filter(lambda el: el in name, invalid.keys()):
            raise Exception("Please use a raw string or escape control code '\\%s'" % invalid[cc])
        sanitized, chars = ([], '')
        for split in name.split():
            for c in split:
                if valid_fn(c):
                    chars += str(c) if c == '_' else c
                else:
                    short = self.shortened_character_name(c, self.eliminations, self.substitutions, self.transforms)
                    sanitized.extend([chars] if chars else [])
                    if short != '':
                        sanitized.append(short)
                    chars = ''
            if chars:
                sanitized.extend([chars])
                chars = ''
        return self._process_underscores(sanitized + ([chars] if chars else []))