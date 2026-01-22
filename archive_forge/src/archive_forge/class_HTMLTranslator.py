import os.path
import docutils
from docutils import frontend, nodes, writers, io
from docutils.transforms import writer_aux
from docutils.writers import _html_base
class HTMLTranslator(writers._html_base.HTMLTranslator):
    """
    This writer generates `polyglot markup`: HTML5 that is also valid XML.

    Safe subclassing: when overriding, treat ``visit_*`` and ``depart_*``
    methods as a unit to prevent breaks due to internal changes. See the
    docstring of docutils.writers._html_base.HTMLTranslator for details
    and examples.
    """

    def visit_acronym(self, node):
        self.body.append(self.starttag(node, 'abbr', ''))

    def depart_acronym(self, node):
        self.body.append('</abbr>')

    def visit_authors(self, node):
        self.visit_docinfo_item(node, 'authors', meta=False)
        for subnode in node:
            self.add_meta('<meta name="author" content="%s" />\n' % self.attval(subnode.astext()))

    def depart_authors(self, node):
        self.depart_docinfo_item()

    def visit_copyright(self, node):
        self.visit_docinfo_item(node, 'copyright', meta=False)
        self.add_meta('<meta name="dcterms.rights" content="%s" />\n' % self.attval(node.astext()))

    def depart_copyright(self, node):
        self.depart_docinfo_item()

    def visit_date(self, node):
        self.visit_docinfo_item(node, 'date', meta=False)
        self.add_meta('<meta name="dcterms.date" content="%s" />\n' % self.attval(node.astext()))

    def depart_date(self, node):
        self.depart_docinfo_item()

    def visit_meta(self, node):
        if node.hasattr('lang'):
            node['xml:lang'] = node['lang']
        meta = self.emptytag(node, 'meta', **node.non_default_attributes())
        self.add_meta(meta)

    def depart_meta(self, node):
        pass

    def visit_organization(self, node):
        self.visit_docinfo_item(node, 'organization', meta=False)

    def depart_organization(self, node):
        self.depart_docinfo_item()