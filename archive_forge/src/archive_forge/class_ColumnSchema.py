import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
class ColumnSchema(object):

    def __init__(self, name, mutable, persistent, type_, extensions={}):
        self.name = name
        self.mutable = mutable
        self.persistent = persistent
        self.type = type_
        self.unique = False
        self.extensions = extensions

    @staticmethod
    def from_json(json, name, allow_extensions=False):
        parser = ovs.db.parser.Parser(json, 'schema for column %s' % name)
        mutable = parser.get_optional('mutable', [bool], True)
        ephemeral = parser.get_optional('ephemeral', [bool], False)
        _types = [str]
        _types.extend([dict])
        type_ = ovs.db.types.Type.from_json(parser.get('type', _types))
        if allow_extensions:
            extensions = parser.get_optional('extensions', [dict], {})
        else:
            extensions = {}
        parser.finish()
        if not mutable and (type_.key.is_weak_ref() or (type_.value and type_.value.is_weak_ref())):
            mutable = True
        return ColumnSchema(name, mutable, not ephemeral, type_, extensions)

    def to_json(self):
        json = {'type': self.type.to_json()}
        if not self.mutable:
            json['mutable'] = False
        if not self.persistent:
            json['ephemeral'] = True
        if self.extensions:
            json['extensions'] = self.extensions
        return json