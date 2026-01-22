import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
class BaseFunction(Callable):
    """
    Base type class for some function types.
    """

    def __init__(self, template):
        if isinstance(template, (list, tuple)):
            self.templates = tuple(template)
            keys = set((temp.key for temp in self.templates))
            if len(keys) != 1:
                raise ValueError('incompatible templates: keys = %s' % (keys,))
            self.typing_key, = keys
        else:
            self.templates = (template,)
            self.typing_key = template.key
        self._impl_keys = {}
        name = '%s(%s)' % (self.__class__.__name__, self.typing_key)
        self._depth = 0
        super(BaseFunction, self).__init__(name)

    @property
    def key(self):
        return (self.typing_key, self.templates)

    def augment(self, other):
        """
        Augment this function type with the other function types' templates,
        so as to support more input types.
        """
        if type(other) is type(self) and other.typing_key == self.typing_key:
            return type(self)(self.templates + other.templates)

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self._impl_keys[sig.args]

    def get_call_type(self, context, args, kws):
        prefer_lit = [True, False]
        prefer_not = [False, True]
        failures = _ResolutionFailures(context, self, args, kws, depth=self._depth)
        from numba.core.target_extension import get_local_target
        target_hw = get_local_target(context)
        order = utils.order_by_target_specificity(target_hw, self.templates, fnkey=self.key[0])
        self._depth += 1
        for temp_cls in order:
            temp = temp_cls(context)
            choice = prefer_lit if temp.prefer_literal else prefer_not
            for uselit in choice:
                try:
                    if uselit:
                        sig = temp.apply(args, kws)
                    else:
                        nolitargs = tuple([_unlit_non_poison(a) for a in args])
                        nolitkws = {k: _unlit_non_poison(v) for k, v in kws.items()}
                        sig = temp.apply(nolitargs, nolitkws)
                except Exception as e:
                    if utils.use_new_style_errors() and (not isinstance(e, errors.NumbaError)):
                        raise e
                    else:
                        sig = None
                        failures.add_error(temp, False, e, uselit)
                else:
                    if sig is not None:
                        self._impl_keys[sig.args] = temp.get_impl_key(sig)
                        self._depth -= 1
                        return sig
                    else:
                        registered_sigs = getattr(temp, 'cases', None)
                        if registered_sigs is not None:
                            msg = 'No match for registered cases:\n%s'
                            msg = msg % '\n'.join((' * {}'.format(x) for x in registered_sigs))
                        else:
                            msg = 'No match.'
                        failures.add_error(temp, True, msg, uselit)
        failures.raise_error()

    def get_call_signatures(self):
        sigs = []
        is_param = False
        for temp in self.templates:
            sigs += getattr(temp, 'cases', [])
            is_param = is_param or hasattr(temp, 'generic')
        return (sigs, is_param)