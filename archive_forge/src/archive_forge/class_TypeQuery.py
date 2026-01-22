from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class TypeQuery(Query):
    """
    Schema query class that searches for Type references in the specified
    schema. Matches on root types only.

    """

    def execute(self, schema):
        if schema.builtin(self.ref):
            name = self.ref[0]
            b = Factory.create(schema, name)
            log.debug('%s, found builtin (%s)', self.id, name)
            return b
        result = schema.types.get(self.ref)
        if self.filter(result):
            result = None
        return self.result(result)