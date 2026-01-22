import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
class IdlSchema(DbSchema):

    def __init__(self, name, version, tables, idlPrefix, idlHeader, cDecls, hDecls):
        DbSchema.__init__(self, name, version, tables)
        self.idlPrefix = idlPrefix
        self.idlHeader = idlHeader
        self.cDecls = cDecls
        self.hDecls = hDecls

    @staticmethod
    def from_json(json):
        parser = ovs.db.parser.Parser(json, 'IDL schema')
        idlPrefix = parser.get('idlPrefix', (str,))
        idlHeader = parser.get('idlHeader', (str,))
        cDecls = parser.get_optional('cDecls', (str,), '')
        hDecls = parser.get_optional('hDecls', (str,), '')
        subjson = dict(json)
        del subjson['idlPrefix']
        del subjson['idlHeader']
        subjson.pop('cDecls', None)
        subjson.pop('hDecls', None)
        schema = DbSchema.from_json(subjson, allow_extensions=True)
        return IdlSchema(schema.name, schema.version, schema.tables, idlPrefix, idlHeader, cDecls, hDecls)