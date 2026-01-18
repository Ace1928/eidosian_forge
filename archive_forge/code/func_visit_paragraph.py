from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.html5 import HTML5Translator as SphinxHTML5Translator
def visit_paragraph(self, node):
    """Visit a paragraph HTML element.

        Replaces implicit headings with an h3 tag and defers to default
        behavior for normal paragraph elements.
        """
    if self.is_implicit_heading(node):
        text = node[0][0]
        self.body.append(f'<h3>{text}</h3>\n')
        raise nodes.SkipNode
    else:
        super().visit_paragraph(node)