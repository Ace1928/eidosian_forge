from __future__ import absolute_import
from ruamel.yaml.reader import Reader
from ruamel.yaml.scanner import Scanner, RoundTripScanner
from ruamel.yaml.parser import Parser, RoundTripParser
from ruamel.yaml.composer import Composer
from ruamel.yaml.constructor import (
from ruamel.yaml.resolver import VersionedResolver
class BaseLoader(Reader, Scanner, Parser, Composer, BaseConstructor, VersionedResolver):

    def __init__(self, stream, version=None, preserve_quotes=None):
        Reader.__init__(self, stream, loader=self)
        Scanner.__init__(self, loader=self)
        Parser.__init__(self, loader=self)
        Composer.__init__(self, loader=self)
        BaseConstructor.__init__(self, loader=self)
        VersionedResolver.__init__(self, version, loader=self)