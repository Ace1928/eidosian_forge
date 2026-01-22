from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class StrictLookup(LookupBase):
    """Strict variable lookup mechanism for expressions.
    
    Referencing an undefined variable using this lookup style will immediately
    raise an ``UndefinedError``:
    
    >>> expr = Expression('nothing', lookup='strict')
    >>> try:
    ...     expr.evaluate({})
    ... except UndefinedError as e:
    ...     print(e.msg)
    "nothing" not defined
    
    The same happens when a non-existing attribute or item is accessed on an
    existing object:
    
    >>> expr = Expression('something.nil', lookup='strict')
    >>> try:
    ...     expr.evaluate({'something': dict()})
    ... except UndefinedError as e:
    ...     print(e.msg)
    {} has no member named "nil"
    """

    @classmethod
    def undefined(cls, key, owner=UNDEFINED):
        """Raise an ``UndefinedError`` immediately."""
        __traceback_hide__ = True
        raise UndefinedError(key, owner=owner)