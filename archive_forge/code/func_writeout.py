from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def writeout(self, outfile):
    for record in self.get_lint():
        outfile.write('{:s}:{}\n'.format(self.infile_path, record))