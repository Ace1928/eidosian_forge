from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
@classmethod
def undefined(cls, key, owner=UNDEFINED):
    """Raise an ``UndefinedError`` immediately."""
    __traceback_hide__ = True
    raise UndefinedError(key, owner=owner)