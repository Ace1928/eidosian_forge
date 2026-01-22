import os
import sys
def _use_c_impl(py_impl, name=None, globs=None):
    """
    Decorator. Given an object implemented in Python, with a name like
    ``Foo``, import the corresponding C implementation from
    ``zope.interface._zope_interface_coptimizations`` with the name
    ``Foo`` and use it instead.

    If the ``PURE_PYTHON`` environment variable is set to any value
    other than ``"0"``, or we're on PyPy, ignore the C implementation
    and return the Python version. If the C implementation cannot be
    imported, return the Python version. If ``PURE_PYTHON`` is set to
    0, *require* the C implementation (let the ImportError propagate);
    note that PyPy can import the C implementation in this case (and all
    tests pass).

    In all cases, the Python version is kept available. in the module
    globals with the name ``FooPy`` and the name ``FooFallback`` (both
    conventions have been used; the C implementation of some functions
    looks for the ``Fallback`` version, as do some of the Sphinx
    documents).

    Example::

        @_use_c_impl
        class Foo(object):
            ...
    """
    name = name or py_impl.__name__
    globs = globs or sys._getframe(1).f_globals

    def find_impl():
        if not _should_attempt_c_optimizations():
            return py_impl
        c_opt = _c_optimizations_available()
        if not c_opt:
            return py_impl
        __traceback_info__ = c_opt
        return getattr(c_opt, name)
    c_impl = find_impl()
    globs[name + 'Py'] = py_impl
    globs[name + 'Fallback'] = py_impl
    return c_impl