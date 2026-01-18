import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def visit_link(self, mdnode):
    ref_node = nodes.reference()
    destination = mdnode.destination
    _, ext = splitext(destination)
    url_check = urlparse(destination)
    known_url_schemes = self.config.get('known_url_schemes')
    if known_url_schemes:
        scheme_known = url_check.scheme in known_url_schemes
    else:
        scheme_known = bool(url_check.scheme)
    if not scheme_known and ext.replace('.', '') in self.supported:
        destination = destination.replace(ext, '')
    ref_node['refuri'] = destination
    ref_node.line = self._get_line(mdnode)
    if mdnode.title:
        ref_node['title'] = mdnode.title
    next_node = ref_node
    if not url_check.fragment and (not scheme_known):
        wrap_node = addnodes.pending_xref(reftarget=unquote(destination), reftype='any', refdomain=None, refexplicit=True, refwarn=True)
        wrap_node.line = self._get_line(mdnode)
        if mdnode.title:
            wrap_node['title'] = mdnode.title
        wrap_node.append(ref_node)
        next_node = wrap_node
    self.current_node.append(next_node)
    self.current_node = ref_node