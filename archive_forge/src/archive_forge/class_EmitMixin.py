import warnings
from twisted.trial import unittest, util
class EmitMixin:
    """
    Mixin for emiting a variety of warnings.
    """

    def _emit(self):
        warnings.warn(METHOD_WARNING_MSG, MethodWarning)
        warnings.warn(CLASS_WARNING_MSG, ClassWarning)
        warnings.warn(MODULE_WARNING_MSG, ModuleWarning)