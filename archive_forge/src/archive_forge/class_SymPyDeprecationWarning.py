import warnings
import contextlib
from textwrap import dedent
class SymPyDeprecationWarning(DeprecationWarning):
    """
    A warning for deprecated features of SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Note that simply constructing this class will not cause a warning to be
    issued. To do that, you must call the :func`sympy_deprecation_warning`
    function. For this reason, it is not recommended to ever construct this
    class directly.

    Explanation
    ===========

    The ``SymPyDeprecationWarning`` class is a subclass of
    ``DeprecationWarning`` that is used for all deprecations in SymPy. A
    special subclass is used so that we can automatically augment the warning
    message with additional metadata about the version the deprecation was
    introduced in and a link to the documentation. This also allows users to
    explicitly filter deprecation warnings from SymPy using ``warnings``
    filters (see :ref:`silencing-sympy-deprecation-warnings`).

    Additionally, ``SymPyDeprecationWarning`` is enabled to be shown by
    default, unlike normal ``DeprecationWarning``\\s, which are only shown by
    default in interactive sessions. This ensures that deprecation warnings in
    SymPy will actually be seen by users.

    See the documentation of :func:`sympy_deprecation_warning` for a
    description of the parameters to this function.

    To mark a function as deprecated, you can use the :func:`@deprecated
    <sympy.utilities.decorator.deprecated>` decorator.

    See Also
    ========
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    """

    def __init__(self, message, *, deprecated_since_version, active_deprecations_target):
        super().__init__(message, deprecated_since_version, active_deprecations_target)
        self.message = message
        if not isinstance(deprecated_since_version, str):
            raise TypeError(f"'deprecated_since_version' should be a string, got {deprecated_since_version!r}")
        self.deprecated_since_version = deprecated_since_version
        self.active_deprecations_target = active_deprecations_target
        if any((i in active_deprecations_target for i in '()=')):
            raise ValueError("active_deprecations_target be the part inside of the '(...)='")
        self.full_message = f'\n\n{dedent(message).strip()}\n\nSee https://docs.sympy.org/latest/explanation/active-deprecations.html#{active_deprecations_target}\nfor details.\n\nThis has been deprecated since SymPy version {deprecated_since_version}. It\nwill be removed in a future version of SymPy.\n'

    def __str__(self):
        return self.full_message

    def __repr__(self):
        return f'{self.__class__.__name__}({self.message!r}, deprecated_since_version={self.deprecated_since_version!r}, active_deprecations_target={self.active_deprecations_target!r})'

    def __eq__(self, other):
        return isinstance(other, SymPyDeprecationWarning) and self.args == other.args

    @classmethod
    def _new(cls, message, deprecated_since_version, active_deprecations_target):
        return cls(message, deprecated_since_version=deprecated_since_version, active_deprecations_target=active_deprecations_target)

    def __reduce__(self):
        return (self._new, (self.message, self.deprecated_since_version, self.active_deprecations_target))