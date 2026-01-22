from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class _OverloadAttributeTemplate(_TemplateTargetHelperMixin, AttributeTemplate):
    """
    A base class of templates for @overload_attribute functions.
    """
    is_method = False

    def __init__(self, context):
        super(_OverloadAttributeTemplate, self).__init__(context)
        self.context = context
        self._init_once()

    def _init_once(self):
        cls = type(self)
        attr = cls._attr
        lower_getattr = self._get_target_registry('attribute').lower_getattr

        @lower_getattr(cls.key, attr)
        def getattr_impl(context, builder, typ, value):
            typingctx = context.typing_context
            fnty = cls._get_function_type(typingctx, typ)
            sig = cls._get_signature(typingctx, fnty, (typ,), {})
            call = context.get_function(fnty, sig)
            return call(builder, (value,))

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None
        fnty = self._get_function_type(self.context, typ)
        sig = self._get_signature(self.context, fnty, (typ,), {})
        for template in fnty.templates:
            self._inline_overloads.update(template._inline_overloads)
        return sig.return_type

    @classmethod
    def _get_signature(cls, typingctx, fnty, args, kws):
        sig = fnty.get_call_type(typingctx, args, kws)
        sig = sig.replace(pysig=utils.pysignature(cls._overload_func))
        return sig

    @classmethod
    def _get_function_type(cls, typingctx, typ):
        return typingctx.resolve_value_type(cls._overload_func)