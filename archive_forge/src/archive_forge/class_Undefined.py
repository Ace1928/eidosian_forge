from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class Undefined(object):
    """Represents a reference to an undefined variable.
    
    Unlike the Python runtime, template expressions can refer to an undefined
    variable without causing a `NameError` to be raised. The result will be an
    instance of the `Undefined` class, which is treated the same as ``False`` in
    conditions, but raise an exception on any other operation:
    
    >>> foo = Undefined('foo')
    >>> bool(foo)
    False
    >>> list(foo)
    []
    >>> print(foo)
    undefined
    
    However, calling an undefined variable, or trying to access an attribute
    of that variable, will raise an exception that includes the name used to
    reference that undefined variable.
    
    >>> try:
    ...     foo('bar')
    ... except UndefinedError as e:
    ...     print(e.msg)
    "foo" not defined

    >>> try:
    ...     foo.bar
    ... except UndefinedError as e:
    ...     print(e.msg)
    "foo" not defined
    
    :see: `LenientLookup`
    """
    __slots__ = ['_name', '_owner']

    def __init__(self, name, owner=UNDEFINED):
        """Initialize the object.
        
        :param name: the name of the reference
        :param owner: the owning object, if the variable is accessed as a member
        """
        self._name = name
        self._owner = owner

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self._name)

    def __str__(self):
        return 'undefined'

    def _die(self, *args, **kwargs):
        """Raise an `UndefinedError`."""
        __traceback_hide__ = True
        raise UndefinedError(self._name, self._owner)
    __call__ = __getattr__ = __getitem__ = _die
    __length_hint__ = None