import re
import docutils
from docutils import nodes, writers, languages
def write_colspecs(self):
    self.body.append('%s.\n' % ('L ' * len(self.colspecs)))