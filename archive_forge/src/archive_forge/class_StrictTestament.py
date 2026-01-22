from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
class StrictTestament(Testament):
    """This testament format is for use as a checksum in bundle format 0.8"""
    long_header = 'bazaar-ng testament version 2.1\n'
    short_header = 'bazaar-ng testament short form 2.1\n'
    include_root = False

    def _entry_to_line(self, path, ie):
        l = Testament._entry_to_line(self, path, ie)[:-1]
        l += ' ' + ie.revision.decode('utf-8')
        l += {True: ' yes\n', False: ' no\n'}[ie.executable]
        return l