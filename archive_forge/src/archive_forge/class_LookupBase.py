from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class LookupBase(object):
    """Abstract base class for variable lookup implementations."""

    @classmethod
    def globals(cls, data):
        """Construct the globals dictionary to use as the execution context for
        the expression or suite.
        """
        return {'__data__': data, '_lookup_name': cls.lookup_name, '_lookup_attr': cls.lookup_attr, '_lookup_item': cls.lookup_item, 'UndefinedError': UndefinedError}

    @classmethod
    def lookup_name(cls, data, name):
        __traceback_hide__ = True
        val = data.get(name, UNDEFINED)
        if val is UNDEFINED:
            val = BUILTINS.get(name, val)
            if val is UNDEFINED:
                val = cls.undefined(name)
        return val

    @classmethod
    def lookup_attr(cls, obj, key):
        __traceback_hide__ = True
        try:
            val = getattr(obj, key)
        except AttributeError:
            if hasattr(obj.__class__, key):
                raise
            else:
                try:
                    val = obj[key]
                except (KeyError, TypeError):
                    val = cls.undefined(key, owner=obj)
        return val

    @classmethod
    def lookup_item(cls, obj, key):
        __traceback_hide__ = True
        if len(key) == 1:
            key = key[0]
        try:
            return obj[key]
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            if isinstance(key, six.string_types):
                val = getattr(obj, key, UNDEFINED)
                if val is UNDEFINED:
                    val = cls.undefined(key, owner=obj)
                return val
            raise

    @classmethod
    def undefined(cls, key, owner=UNDEFINED):
        """Can be overridden by subclasses to specify behavior when undefined
        variables are accessed.
        
        :param key: the name of the variable
        :param owner: the owning object, if the variable is accessed as a member
        """
        raise NotImplementedError