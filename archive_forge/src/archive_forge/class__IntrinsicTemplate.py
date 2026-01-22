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
class _IntrinsicTemplate(_TemplateTargetHelperMixin, AbstractTemplate):
    """
    A base class of templates for intrinsic definition
    """

    def generic(self, args, kws):
        """
        Type the intrinsic by the arguments.
        """
        lower_builtin = self._get_target_registry('intrinsic').lower
        cache_key = (self.context, args, tuple(kws.items()))
        try:
            return self._impl_cache[cache_key]
        except KeyError:
            pass
        result = self._definition_func(self.context, *args, **kws)
        if result is None:
            return
        [sig, imp] = result
        pysig = utils.pysignature(self._definition_func)
        parameters = list(pysig.parameters.values())[1:]
        sig = sig.replace(pysig=pysig.replace(parameters=parameters))
        self._impl_cache[cache_key] = sig
        self._overload_cache[sig.args] = imp
        lower_builtin(imp, *sig.args)(imp)
        return sig

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        return self._overload_cache[sig.args]

    def get_template_info(self):
        basepath = os.path.dirname(os.path.dirname(numba.__file__))
        impl = self._definition_func
        code, firstlineno, path = self.get_source_code_info(impl)
        sig = str(utils.pysignature(impl))
        info = {'kind': 'intrinsic', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
        return info