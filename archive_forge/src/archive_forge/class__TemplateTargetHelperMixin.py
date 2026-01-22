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
class _TemplateTargetHelperMixin(object):
    """Mixin for helper methods that assist with target/registry resolution"""

    def _get_target_registry(self, reason):
        """Returns the registry for the current target.

        Parameters
        ----------
        reason: str
            Reason for the resolution. Expects a noun.
        Returns
        -------
        reg : a registry suitable for the current target.
        """
        from numba.core.target_extension import _get_local_target_checked, dispatcher_registry
        hwstr = self.metadata.get('target', 'generic')
        target_hw = _get_local_target_checked(self.context, hwstr, reason)
        disp = dispatcher_registry[target_hw]
        tgtctx = disp.targetdescr.target_context
        tgtctx.refresh()
        if builtin_registry in tgtctx._registries:
            reg = builtin_registry
        else:
            registries = iter(tgtctx._registries)
            reg = next(registries)
        return reg