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
class BoundFunction(Callable, Opaque):
    """
    A function with an implicit first argument (denoted as *this* below).
    """

    def __init__(self, template, this):
        newcls = type(template.__name__ + '.' + str(this), (template,), dict(this=this))
        self.template = newcls
        self.typing_key = self.template.key
        self.this = this
        name = '%s(%s for %s)' % (self.__class__.__name__, self.typing_key, self.this)
        super(BoundFunction, self).__init__(name)

    def unify(self, typingctx, other):
        if isinstance(other, BoundFunction) and self.typing_key == other.typing_key:
            this = typingctx.unify_pairs(self.this, other.this)
            if this is not None:
                return self.copy(this=this)

    def copy(self, this):
        return type(self)(self.template, this)

    @property
    def key(self):
        unique_impl = getattr(self.template, '_overload_func', None)
        return (self.typing_key, self.this, unique_impl)

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self.typing_key

    def get_call_type(self, context, args, kws):
        template = self.template(context)
        literal_e = None
        nonliteral_e = None
        out = None
        choice = [True, False] if template.prefer_literal else [False, True]
        for uselit in choice:
            if uselit:
                try:
                    out = template.apply(args, kws)
                except Exception as exc:
                    if utils.use_new_style_errors() and (not isinstance(exc, errors.NumbaError)):
                        raise exc
                    if isinstance(exc, errors.ForceLiteralArg):
                        raise exc
                    literal_e = exc
                    out = None
                else:
                    break
            else:
                unliteral_args = tuple([_unlit_non_poison(a) for a in args])
                unliteral_kws = {k: _unlit_non_poison(v) for k, v in kws.items()}
                skip = unliteral_args == args and kws == unliteral_kws
                if not skip and out is None:
                    try:
                        out = template.apply(unliteral_args, unliteral_kws)
                    except Exception as exc:
                        if isinstance(exc, errors.ForceLiteralArg):
                            if template.prefer_literal:
                                raise exc
                        nonliteral_e = exc
                    else:
                        break
        if out is None and (nonliteral_e is not None or literal_e is not None):
            header = '- Resolution failure for {} arguments:\n{}\n'
            tmplt = _termcolor.highlight(header)
            if config.DEVELOPER_MODE:
                indent = ' ' * 4

                def add_bt(error):
                    if isinstance(error, BaseException):
                        bt = traceback.format_exception(type(error), error, error.__traceback__)
                    else:
                        bt = ['']
                    nd2indent = '\n{}'.format(2 * indent)
                    errstr = _termcolor.reset(nd2indent + nd2indent.join(_bt_as_lines(bt)))
                    return _termcolor.reset(errstr)
            else:
                add_bt = lambda X: ''

            def nested_msg(literalness, e):
                estr = str(e)
                estr = estr if estr else str(repr(e)) + add_bt(e)
                new_e = errors.TypingError(textwrap.dedent(estr))
                return tmplt.format(literalness, str(new_e))
            raise errors.TypingError(nested_msg('literal', literal_e) + nested_msg('non-literal', nonliteral_e))
        return out

    def get_call_signatures(self):
        sigs = getattr(self.template, 'cases', [])
        is_param = hasattr(self.template, 'generic')
        return (sigs, is_param)