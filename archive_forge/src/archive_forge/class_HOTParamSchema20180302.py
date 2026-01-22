from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
class HOTParamSchema20180302(HOTParamSchema20170224):
    KEYS_20180302 = TAGS, = ('tags',)
    KEYS = HOTParamSchema20170224.KEYS + KEYS_20180302
    PARAMETER_KEYS = KEYS

    @classmethod
    def from_dict(cls, param_name, schema_dict):
        """Return a Parameter Schema object from a legacy schema dictionary.

        :param param_name: name of the parameter owning the schema; used
               for more verbose logging
        :type  param_name: str
        """
        cls._validate_dict(param_name, schema_dict)
        return cls(schema_dict[cls.TYPE], description=schema_dict.get(HOTParamSchema.DESCRIPTION), default=schema_dict.get(HOTParamSchema.DEFAULT), constraints=list(cls._constraints(param_name, schema_dict)), hidden=schema_dict.get(HOTParamSchema.HIDDEN, False), label=schema_dict.get(HOTParamSchema.LABEL), immutable=schema_dict.get(HOTParamSchema.IMMUTABLE, False), tags=schema_dict.get(HOTParamSchema20180302.TAGS))