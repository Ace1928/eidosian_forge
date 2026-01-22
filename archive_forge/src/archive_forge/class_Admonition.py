from docutils.parsers.rst import Directive
from docutils.parsers.rst import states, directives
from docutils.parsers.rst.roles import set_classes
from docutils import nodes
class Admonition(BaseAdmonition):
    required_arguments = 1
    node_class = nodes.admonition