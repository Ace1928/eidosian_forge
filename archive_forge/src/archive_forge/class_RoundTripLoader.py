from __future__ import absolute_import
from ruamel.yaml.reader import Reader
from ruamel.yaml.scanner import Scanner, RoundTripScanner
from ruamel.yaml.parser import Parser, RoundTripParser
from ruamel.yaml.composer import Composer
from ruamel.yaml.constructor import (
from ruamel.yaml.resolver import VersionedResolver
class RoundTripLoader(Reader, RoundTripScanner, RoundTripParser, Composer, RoundTripConstructor, VersionedResolver):

    def __init__(self, stream, version=None, preserve_quotes=None):
        Reader.__init__(self, stream, loader=self)
        RoundTripScanner.__init__(self, loader=self)
        RoundTripParser.__init__(self, loader=self)
        Composer.__init__(self, loader=self)
        RoundTripConstructor.__init__(self, preserve_quotes=preserve_quotes, loader=self)
        VersionedResolver.__init__(self, version, loader=self)