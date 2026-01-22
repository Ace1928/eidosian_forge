import inspect
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from stevedore import extension
def _detailed_list(mgr, over='', under='-', titlecase=False):
    for name in sorted(mgr.names()):
        ext = mgr[name]
        if over:
            yield (over * len(ext.name), ext.module_name)
        if titlecase:
            yield (ext.name.title(), ext.module_name)
        else:
            yield (ext.name, ext.module_name)
        if under:
            yield (under * len(ext.name), ext.module_name)
        yield ('\n', ext.module_name)
        doc = _get_docstring(ext.plugin)
        if doc:
            yield (doc, ext.module_name)
        else:
            yield ('.. warning:: No documentation found for {} in {}'.format(ext.name, ext.entry_point_target), ext.module_name)
        yield ('\n', ext.module_name)