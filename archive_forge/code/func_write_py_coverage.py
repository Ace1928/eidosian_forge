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
def write_py_coverage(self) -> None:
    output_file = path.join(self.outdir, 'python.txt')
    failed = []
    with open(output_file, 'w', encoding='utf-8') as op:
        if self.config.coverage_write_headline:
            write_header(op, 'Undocumented Python objects', '=')
        keys = sorted(self.py_undoc.keys())
        for name in keys:
            undoc = self.py_undoc[name]
            if 'error' in undoc:
                failed.append((name, undoc['error']))
            else:
                if not undoc['classes'] and (not undoc['funcs']):
                    continue
                write_header(op, name)
                if undoc['funcs']:
                    op.write('Functions:\n')
                    op.writelines((' * %s\n' % x for x in undoc['funcs']))
                    if self.config.coverage_show_missing_items:
                        if self.app.quiet or self.app.warningiserror:
                            for func in undoc['funcs']:
                                logger.warning(__('undocumented python function: %s :: %s'), name, func)
                        else:
                            for func in undoc['funcs']:
                                logger.info(red('undocumented  ') + 'py  ' + 'function  ' + '%-30s' % func + red(' - in module ') + name)
                    op.write('\n')
                if undoc['classes']:
                    op.write('Classes:\n')
                    for class_name, methods in sorted(undoc['classes'].items()):
                        if not methods:
                            op.write(' * %s\n' % class_name)
                            if self.config.coverage_show_missing_items:
                                if self.app.quiet or self.app.warningiserror:
                                    logger.warning(__('undocumented python class: %s :: %s'), name, class_name)
                                else:
                                    logger.info(red('undocumented  ') + 'py  ' + 'class     ' + '%-30s' % class_name + red(' - in module ') + name)
                        else:
                            op.write(' * %s -- missing methods:\n\n' % class_name)
                            op.writelines(('   - %s\n' % x for x in methods))
                            if self.config.coverage_show_missing_items:
                                if self.app.quiet or self.app.warningiserror:
                                    for meth in methods:
                                        logger.warning(__('undocumented python method:' + ' %s :: %s :: %s'), name, class_name, meth)
                                else:
                                    for meth in methods:
                                        logger.info(red('undocumented  ') + 'py  ' + 'method    ' + '%-30s' % (class_name + '.' + meth) + red(' - in module ') + name)
                    op.write('\n')
        if failed:
            write_header(op, 'Modules that failed to import')
            op.writelines((' * %s -- %s\n' % x for x in failed))