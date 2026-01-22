important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class Param(PythonBaseNode):
    """
    It's a helper class that makes business logic with params much easier. The
    Python grammar defines no ``param`` node. It defines it in a different way
    that is not really suited to working with parameters.
    """
    type = 'param'

    def __init__(self, children, parent=None):
        super().__init__(children)
        self.parent = parent

    @property
    def star_count(self):
        """
        Is `0` in case of `foo`, `1` in case of `*foo` or `2` in case of
        `**foo`.
        """
        first = self.children[0]
        if first in ('*', '**'):
            return len(first.value)
        return 0

    @property
    def default(self):
        """
        The default is the test node that appears after the `=`. Is `None` in
        case no default is present.
        """
        has_comma = self.children[-1] == ','
        try:
            if self.children[-2 - int(has_comma)] == '=':
                return self.children[-1 - int(has_comma)]
        except IndexError:
            return None

    @property
    def annotation(self):
        """
        The default is the test node that appears after `:`. Is `None` in case
        no annotation is present.
        """
        tfpdef = self._tfpdef()
        if tfpdef.type == 'tfpdef':
            assert tfpdef.children[1] == ':'
            assert len(tfpdef.children) == 3
            annotation = tfpdef.children[2]
            return annotation
        else:
            return None

    def _tfpdef(self):
        """
        tfpdef: see e.g. grammar36.txt.
        """
        offset = int(self.children[0] in ('*', '**'))
        return self.children[offset]

    @property
    def name(self):
        """
        The `Name` leaf of the param.
        """
        if self._tfpdef().type == 'tfpdef':
            return self._tfpdef().children[0]
        else:
            return self._tfpdef()

    def get_defined_names(self, include_setitem=False):
        return [self.name]

    @property
    def position_index(self):
        """
        Property for the positional index of a paramter.
        """
        index = self.parent.children.index(self)
        try:
            keyword_only_index = self.parent.children.index('*')
            if index > keyword_only_index:
                index -= 2
        except ValueError:
            pass
        try:
            keyword_only_index = self.parent.children.index('/')
            if index > keyword_only_index:
                index -= 2
        except ValueError:
            pass
        return index - 1

    def get_parent_function(self):
        """
        Returns the function/lambda of a parameter.
        """
        return self.search_ancestor('funcdef', 'lambdef')

    def get_code(self, include_prefix=True, include_comma=True):
        """
        Like all the other get_code functions, but includes the param
        `include_comma`.

        :param include_comma bool: If enabled includes the comma in the string output.
        """
        if include_comma:
            return super().get_code(include_prefix)
        children = self.children
        if children[-1] == ',':
            children = children[:-1]
        return self._get_code_for_children(children, include_prefix=include_prefix)

    def __repr__(self):
        default = '' if self.default is None else '=%s' % self.default.get_code()
        return '<%s: %s>' % (type(self).__name__, str(self._tfpdef()) + default)