from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class LenientLookup(LookupBase):
    """Default variable lookup mechanism for expressions.
    
    When an undefined variable is referenced using this lookup style, the
    reference evaluates to an instance of the `Undefined` class:
    
    >>> expr = Expression('nothing', lookup='lenient')
    >>> undef = expr.evaluate({})
    >>> undef
    <Undefined 'nothing'>
    
    The same will happen when a non-existing attribute or item is accessed on
    an existing object:
    
    >>> expr = Expression('something.nil', lookup='lenient')
    >>> expr.evaluate({'something': dict()})
    <Undefined 'nil'>
    
    See the documentation of the `Undefined` class for details on the behavior
    of such objects.
    
    :see: `StrictLookup`
    """

    @classmethod
    def undefined(cls, key, owner=UNDEFINED):
        """Return an ``Undefined`` object."""
        __traceback_hide__ = True
        return Undefined(key, owner=owner)