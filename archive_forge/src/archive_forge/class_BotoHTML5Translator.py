from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.html5 import HTML5Translator as SphinxHTML5Translator
class BotoHTML5Translator(SphinxHTML5Translator):
    """Extension of Sphinx's ``HTML5Translator`` for Botocore documentation."""
    IGNORE_IMPLICIT_HEADINGS = ['[REQUIRED]']

    def visit_admonition(self, node, name=''):
        """Uses the h3 tag for admonition titles instead of the p tag."""
        self.body.append(self.starttag(node, 'div', CLASS='admonition ' + name))
        if name:
            title = f"<h3 class='admonition-title'> {admonitionlabels[name]}</h3>"
            self.body.append(title)

    def is_implicit_heading(self, node):
        """Determines if a node is an implicit heading.

        An implicit heading is represented by a paragraph node whose only
        child is a strong node with text that isnt in `IGNORE_IMPLICIT_HEADINGS`.
        """
        return len(node) == 1 and isinstance(node[0], nodes.strong) and (len(node[0]) == 1) and isinstance(node[0][0], nodes.Text) and (node[0][0].astext() not in self.IGNORE_IMPLICIT_HEADINGS)

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