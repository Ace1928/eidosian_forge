from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
class StrictTestament3(StrictTestament):
    """This testament format is for use as a checksum in bundle format 0.9+

    It differs from StrictTestament by including data about the tree root.
    """
    long_header = 'bazaar testament version 3 strict\n'
    short_header = 'bazaar testament short form 3 strict\n'
    include_root = True

    def _escape_path(self, path):
        if contains_linebreaks(path):
            raise ValueError(path)
        if not isinstance(path, str):
            path = path.decode('ascii')
        if path == '':
            path = '.'
        return path.replace('\\', '/').replace(' ', '\\ ')