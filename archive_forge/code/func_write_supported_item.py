import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def write_supported_item(self, modname, itemname, typename, explained, sources, alias):
    self.print('.. function:: {}.{}'.format(modname, itemname))
    self.print('   :noindex:')
    self.print()
    if alias:
        self.print('   Alias to: ``{}``'.format(alias))
    self.print()
    for tcls, source in sources.items():
        if source:
            impl = source['name']
            sig = source['sig']
            filename = source['filename']
            lines = source['lines']
            source_link = github_url.format(commit=commit, path=filename, firstline=lines[0], lastline=lines[1])
            self.print('   - defined by ``{}{}`` at `{}:{}-{} <{}>`_'.format(impl, sig, filename, lines[0], lines[1], source_link))
        else:
            self.print('   - defined by ``{}``'.format(str(tcls)))
    self.print()