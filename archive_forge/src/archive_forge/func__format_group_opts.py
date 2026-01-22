from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
import oslo_i18n
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
from sphinx.util.nodes import nested_parse_with_titles
from oslo_config import cfg
from oslo_config import generator
def _format_group_opts(namespace, group_name, group_obj, opt_list):
    group_name = group_name or 'DEFAULT'
    LOG.debug('%s %s', namespace, group_name)
    for line in _format_group(namespace, group_name, group_obj):
        yield line
    for opt in opt_list:
        for line in _format_opt(opt, group_name):
            yield line