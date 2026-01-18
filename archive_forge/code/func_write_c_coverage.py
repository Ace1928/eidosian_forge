import glob
import inspect
import pickle
import re
from importlib import import_module
from os import path
from typing import IO, Any, Dict, List, Pattern, Set, Tuple
import sphinx
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import red  # type: ignore
from sphinx.util.inspect import safe_getattr
def write_c_coverage(self) -> None:
    output_file = path.join(self.outdir, 'c.txt')
    with open(output_file, 'w', encoding='utf-8') as op:
        if self.config.coverage_write_headline:
            write_header(op, 'Undocumented C API elements', '=')
        op.write('\n')
        for filename, undoc in self.c_undoc.items():
            write_header(op, filename)
            for typ, name in sorted(undoc):
                op.write(' * %-50s [%9s]\n' % (name, typ))
                if self.config.coverage_show_missing_items:
                    if self.app.quiet or self.app.warningiserror:
                        logger.warning(__('undocumented c api: %s [%s] in file %s'), name, typ, filename)
                    else:
                        logger.info(red('undocumented  ') + 'c   ' + 'api       ' + '%-30s' % (name + ' [%9s]' % typ) + red(' - in file ') + filename)
            op.write('\n')