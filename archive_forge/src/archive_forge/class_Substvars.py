import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
class Substvars(_Substvars_Base['Substvars']):
    """Substvars is a dict-like object containing known substvars for a given package.

    >>> substvars = Substvars()
    >>> substvars['foo'] = 'bar, golf'
    >>> substvars['foo']
    'bar, golf'
    >>> substvars.add_dependency('foo', 'dpkg (>= 1.20.0)')
    >>> substvars['foo']
    'bar, dpkg (>= 1.20.0), golf'
    >>> 'foo' in substvars
    True
    >>> sorted(substvars)
    ['foo']
    >>> del substvars['foo']
    >>> substvars['foo']
    Traceback (most recent call last):
        ...
    KeyError: 'foo'
    >>> substvars.get('foo')
    >>> # None
    >>> substvars['foo'] = ""
    >>> substvars['foo']
    ''

    The Substvars object also provide methods for serializing and deserializing
    the substvars into and from the format used by dpkg-gencontrol.

    The Substvars object can be used as a context manager, which causes the substvars
    to be saved when the context manager exits successfully (i.e., no exceptions are raised).
    """
    __slots__ = ['_vars_dict', '_substvars_path']

    def __init__(self):
        self._vars_dict = OrderedDict()
        self._substvars_path = None

    @classmethod
    def load_from_path(cls, substvars_path, missing_ok=False):
        """Shorthand for initializing a Substvars from a file

        The return substvars will have `substvars_path` set to the provided path enabling
        `save()` to work out of the box. This also makes it easy to combine this with the
        context manager interface to automatically save the file again.

        >>> import os
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmpdir:
        ...    filename = os.path.join(tmpdir, "foo.substvars")
        ...    # Obviously, this does not exist
        ...    print("Exists before: " + str(os.path.exists(filename)))
        ...    with Substvars.load_from_path(filename, missing_ok=True) as svars:
        ...        svars.add_dependency("misc:Depends", "bar (>= 1.0)")
        ...    print("Exists after: " + str(os.path.exists(filename)))
        Exists before: False
        Exists after: True

        :param substvars_path: The path to load from
        :param missing_ok: If True, then the path does not have to exist (i.e.
          FileNotFoundError causes an empty Substvars object to be returned).  Combined
          with the context manager, this is useful for packaging helpers that want to
          append / update to the existing if it exists or create it if it does not exist.
        """
        substvars = cls()
        try:
            with open(substvars_path, 'r', encoding='utf-8') as fd:
                substvars.read_substvars(fd)
        except OSError as e:
            if e.errno != errno.ENOENT or not missing_ok:
                raise
        substvars.substvars_path = substvars_path
        return substvars

    @property
    def _vars(self):
        return self._vars_dict

    @_vars.setter
    def _vars(self, vars_dict):
        self._vars_dict = vars_dict

    @property
    def substvars_path(self):
        return self._substvars_path

    @substvars_path.setter
    def substvars_path(self, new_path):
        self._substvars_path = new_path

    def add_dependency(self, substvar, dependency_clause):
        """Add a dependency clause to a given substvar

        >>> substvars = Substvars()
        >>> # add_dependency automatically creates variables
        >>> 'misc:Recommends' not in substvars
        True
        >>> substvars.add_dependency('misc:Recommends', "foo (>= 1.0)")
        >>> substvars['misc:Recommends']
        'foo (>= 1.0)'
        >>> # It can be appended to other variables
        >>> substvars['foo'] = 'bar, golf'
        >>> substvars.add_dependency('foo', 'dpkg (>= 1.20.0)')
        >>> substvars['foo']
        'bar, dpkg (>= 1.20.0), golf'
        >>> # Exact duplicates are ignored
        >>> substvars.add_dependency('foo', 'dpkg (>= 1.20.0)')
        >>> substvars['foo']
        'bar, dpkg (>= 1.20.0), golf'

        """
        try:
            variable = self._vars[substvar]
        except KeyError:
            variable = Substvar()
            self._vars[substvar] = variable
        variable.add_dependency(dependency_clause)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return iter(self._vars)

    def __len__(self) -> int:
        return len(self._vars_dict)

    def __contains__(self, item):
        return item in self._vars

    def __getitem__(self, key):
        return self._vars[key].resolve()

    def __delitem__(self, key):
        del self._vars[key]

    def __setitem__(self, key, value):
        self._vars[key] = Substvar(value)

    @property
    def as_substvar(self):
        """Provides a mapping to the Substvars object for more advanced operations

        Treating a substvars file mostly as a "str -> str" mapping is sufficient for many cases.
        But when full control over the substvars (like fiddling with the assignment operator) is
        needed this attribute is useful.

        >>> content = '''
        ... # Some comment (which is allowed but no one uses them - also, they are not preserved)
        ... shlib:Depends=foo (>= 1.0), libbar2 (>= 2.1-3~)
        ... random:substvar?=With the new assignment operator from dpkg 1.21.8
        ... '''
        >>> substvars = Substvars()
        >>> substvars.read_substvars(content.splitlines())
        >>> substvars.as_substvar["shlib:Depends"].assignment_operator
        '='
        >>> substvars.as_substvar["random:substvar"].assignment_operator
        '?='
        >>> # Mutation is also possible
        >>> substvars.as_substvar["shlib:Depends"].assignment_operator = '?='
        >>> print(substvars.dump(), end="")
        shlib:Depends?=foo (>= 1.0), libbar2 (>= 2.1-3~)
        random:substvar?=With the new assignment operator from dpkg 1.21.8
        """
        return self._vars

    def __eq__(self, other):
        if other is None or not isinstance(other, Substvars):
            return False
        return self._vars == other._vars

    def dump(self):
        """Debug aid that generates a string representation of the content

        For persisting the contents, please consider `save()` or `write_substvars`.
        """
        return ''.join(('{}{}{}\n'.format(k, v.assignment_operator, v.resolve()) for k, v in self._vars.items()))

    def save(self):
        """Save the substvars file

        Replace the path denoted by the `substvars_path` attribute with the
        in-memory version of the substvars.  Note that the `substvars_path`
        property must be not None for this method to work.
        """
        if self._substvars_path is None:
            raise TypeError('The substvar does not have a substvars_path: Please set substvars_path first or use write_substvars')
        with open(self._substvars_path, 'w', encoding='utf-8') as fd:
            return self.write_substvars(fd)

    def write_substvars(self, fileobj):
        """Write a copy of the substvars to an open text file

        :param fileobj: The open file (should open in text mode using the UTF-8 encoding)
        """
        fileobj.writelines(('{}{}{}\n'.format(k, v.assignment_operator, v.resolve()) for k, v in self._vars.items()))

    def read_substvars(self, fileobj):
        """Read substvars from an open text file in the format supported by dpkg-gencontrol

        On success, all existing variables will be discarded and only variables
        from the file will be present after this method completes.  In case of
        any IO related errors, the object retains its state prior to the call
        of this method.

        >>> content = '''
        ... # Some comment (which is allowed but no one uses them - also, they are not preserved)
        ... shlib:Depends=foo (>= 1.0), libbar2 (>= 2.1-3~)
        ... random:substvar?=With the new assignment operator from dpkg 1.21.8
        ... '''
        >>> substvars = Substvars()
        >>> substvars.read_substvars(content.splitlines())
        >>> substvars["shlib:Depends"]
        'foo (>= 1.0), libbar2 (>= 2.1-3~)'
        >>> substvars["random:substvar"]
        'With the new assignment operator from dpkg 1.21.8'

        :param fileobj: An open file (in text mode using the UTF-8 encoding) or an
          iterable of str that provides line by line content.
        """
        vars_dict = OrderedDict()
        for line in fileobj:
            if line.strip() == '' or line[0] == '#':
                continue
            m = _SUBSTVAR_PATTERN.match(line.rstrip('\r\n'))
            if not m:
                continue
            varname, assignment_operator, value = m.groups()
            vars_dict[varname] = Substvar(value, assignment_operator=assignment_operator)
        self._vars = vars_dict