import sys
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.roles import set_classes
from docutils.utils.code_analyzer import Lexer, LexerError, NumberLines
class MathBlock(Directive):
    option_spec = {'class': directives.class_option, 'name': directives.unchanged}
    has_content = True

    def run(self):
        set_classes(self.options)
        self.assert_has_content()
        content = '\n'.join(self.content).split('\n\n')
        _nodes = []
        for block in content:
            if not block:
                continue
            node = nodes.math_block(self.block_text, block, **self.options)
            node.line = self.content_offset + 1
            self.add_name(node)
            _nodes.append(node)
        return _nodes