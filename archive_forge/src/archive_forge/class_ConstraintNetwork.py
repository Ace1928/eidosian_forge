import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
class ConstraintNetwork(object):
    """
    TODO: It is possible to optimize constraint propagation to consider only
          dirty type variables.
    """

    def __init__(self):
        self.constraints = []

    def append(self, constraint):
        self.constraints.append(constraint)

    def propagate(self, typeinfer):
        """
        Execute all constraints.  Errors are caught and returned as a list.
        This allows progressing even though some constraints may fail
        due to lack of information
        (e.g. imprecise types such as List(undefined)).
        """
        errors = []
        for constraint in self.constraints:
            loc = constraint.loc
            with typeinfer.warnings.catch_warnings(filename=loc.filename, lineno=loc.line):
                try:
                    constraint(typeinfer)
                except ForceLiteralArg as e:
                    errors.append(e)
                except TypingError as e:
                    _logger.debug('captured error', exc_info=e)
                    new_exc = TypingError(str(e), loc=constraint.loc, highlighting=False)
                    errors.append(utils.chain_exception(new_exc, e))
                except Exception as e:
                    if utils.use_old_style_errors():
                        _logger.debug('captured error', exc_info=e)
                        msg = 'Internal error at {con}.\n{err}\nEnable logging at debug level for details.'
                        new_exc = TypingError(msg.format(con=constraint, err=str(e)), loc=constraint.loc, highlighting=False)
                        errors.append(utils.chain_exception(new_exc, e))
                    elif utils.use_new_style_errors():
                        raise e
                    else:
                        msg = f"Unknown CAPTURED_ERRORS style: '{config.CAPTURED_ERRORS}'."
                        assert 0, msg
        return errors