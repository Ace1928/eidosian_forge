from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
class HOTParamSchema20170224(HOTParamSchema):

    @classmethod
    def _constraint_from_def(cls, constraint):
        desc = constraint.get(DESCRIPTION)
        if MODULO in constraint:
            cdef = constraint.get(MODULO)
            cls._check_dict(cdef, MODULO_KEYS, 'modulo constraint')
            return constr.Modulo(parameters.Schema.get_num(STEP, cdef), parameters.Schema.get_num(OFFSET, cdef), desc)
        else:
            return super(HOTParamSchema20170224, cls)._constraint_from_def(constraint)