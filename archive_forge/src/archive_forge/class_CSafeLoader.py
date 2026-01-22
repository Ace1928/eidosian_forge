from __future__ import absolute_import
from _ruamel_yaml import CParser, CEmitter  # type: ignore
from ruamel.yaml.constructor import Constructor, BaseConstructor, SafeConstructor
from ruamel.yaml.representer import Representer, SafeRepresenter, BaseRepresenter
from ruamel.yaml.resolver import Resolver, BaseResolver
class CSafeLoader(CParser, SafeConstructor, Resolver):

    def __init__(self, stream, version=None, preserve_quotes=None):
        CParser.__init__(self, stream)
        self._parser = self._composer = self
        SafeConstructor.__init__(self, loader=self)
        Resolver.__init__(self, loadumper=self)