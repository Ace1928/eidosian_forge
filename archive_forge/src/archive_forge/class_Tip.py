from docutils.parsers.rst import Directive
from docutils.parsers.rst import states, directives
from docutils.parsers.rst.roles import set_classes
from docutils import nodes
class Tip(BaseAdmonition):
    node_class = nodes.tip