import sys
import re
import functools
import os
import contextlib
import warnings
import inspect
import pathlib
from typing import Any, Callable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.utilities.exceptions import ignore_warnings # noqa:F401
@contextlib.contextmanager
def warns(warningcls, *, match='', test_stacklevel=True):
    """
    Like raises but tests that warnings are emitted.

    >>> from sympy.testing.pytest import warns
    >>> import warnings

    >>> with warns(UserWarning):
    ...     warnings.warn('deprecated', UserWarning, stacklevel=2)

    >>> with warns(UserWarning):
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type UserWarning    was emitted. The list of emitted warnings is: [].

    ``test_stacklevel`` makes it check that the ``stacklevel`` parameter to
    ``warn()`` is set so that the warning shows the user line of code (the
    code under the warns() context manager). Set this to False if this is
    ambiguous or if the context manager does not test the direct user code
    that emits the warning.

    If the warning is a ``SymPyDeprecationWarning``, this additionally tests
    that the ``active_deprecations_target`` is a real target in the
    ``active-deprecations.md`` file.

    """
    with warnings.catch_warnings(record=True) as warnrec:
        warnings.simplefilter('error')
        warnings.filterwarnings('always', category=warningcls)
        yield warnrec
    if not any((issubclass(w.category, warningcls) for w in warnrec)):
        msg = 'Failed: DID NOT WARN. No warnings of type %s was emitted. The list of emitted warnings is: %s.' % (warningcls, [w.message for w in warnrec])
        raise Failed(msg)
    for w in warnrec:
        assert issubclass(w.category, warningcls)
        if not re.compile(match, re.I).match(str(w.message)):
            raise Failed(f'Failed: WRONG MESSAGE. A warning with of the correct category ({warningcls.__name__}) was issued, but it did not match the given match regex ({match!r})')
    if test_stacklevel:
        for f in inspect.stack():
            thisfile = f.filename
            file = os.path.split(thisfile)[1]
            if file.startswith('test_'):
                break
            elif file == 'doctest.py':
                return
        else:
            raise RuntimeError('Could not find the file for the given warning to test the stacklevel')
        for w in warnrec:
            if w.filename != thisfile:
                msg = f'Failed: Warning has the wrong stacklevel. The warning stacklevel needs to be\nset so that the line of code shown in the warning message is user code that\ncalls the deprecated code (the current stacklevel is showing code from\n{w.filename} (line {w.lineno}), expected {thisfile})'.replace('\n', ' ')
                raise Failed(msg)
    if warningcls == SymPyDeprecationWarning:
        this_file = pathlib.Path(__file__)
        active_deprecations_file = this_file.parent.parent.parent / 'doc' / 'src' / 'explanation' / 'active-deprecations.md'
        if not active_deprecations_file.exists():
            return
        targets = []
        for w in warnrec:
            targets.append(w.message.active_deprecations_target)
        with open(active_deprecations_file, encoding='utf-8') as f:
            text = f.read()
        for target in targets:
            if f'({target})=' not in text:
                raise Failed(f'The active deprecations target {target!r} does not appear to be a valid target in the active-deprecations.md file ({active_deprecations_file}).')