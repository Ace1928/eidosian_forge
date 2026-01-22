from __future__ import absolute_import
from .. import (
from ..helpers import (
import stat
def _dump_stats_group(self, title, items, normal_formatter=None, verbose_formatter=None):
    """Dump a statistics group.

        In verbose mode, do so as a config file so
        that other processors can load the information if they want to.
        :param normal_formatter: the callable to apply to the value
          before displaying it in normal mode
        :param verbose_formatter: the callable to apply to the value
          before displaying it in verbose mode
        """
    if self.verbose:
        self.outf.write('[%s]\n' % (title,))
        for name, value in items:
            if verbose_formatter is not None:
                value = verbose_formatter(value)
            if type(name) == str:
                name = name.replace(' ', '-')
            self.outf.write('%s = %s\n' % (name, value))
        self.outf.write('\n')
    else:
        self.outf.write('%s:\n' % (title,))
        for name, value in items:
            if normal_formatter is not None:
                value = normal_formatter(value)
            self.outf.write('\t%s\t%s\n' % (value, name))