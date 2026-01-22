import re
from .. import osutils
from ..iterablefile import IterableFile
class RioWriter:

    def __init__(self, to_file):
        self._soft_nl = False
        self._to_file = to_file

    def write_stanza(self, stanza):
        if self._soft_nl:
            self._to_file.write(b'\n')
        stanza.write(self._to_file)
        self._soft_nl = True