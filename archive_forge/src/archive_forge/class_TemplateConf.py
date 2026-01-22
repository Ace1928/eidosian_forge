from typing import Union, Optional, Mapping, Dict, Tuple, Iterator
from lark import Tree, Transformer
from lark.exceptions import MissingVariableError
class TemplateConf:
    """Template Configuration

    Allows customization for different uses of Template

    parse() must return a Tree instance.
    """

    def __init__(self, parse=None):
        self._parse = parse

    def test_var(self, var: Union[Tree[str], str]) -> Optional[str]:
        """Given a tree node, if it is a template variable return its name. Otherwise, return None.

        This method may be overridden for customization

        Parameters:
            var: Tree | str - The tree node to test

        """
        if isinstance(var, str):
            return _get_template_name(var)
        if isinstance(var, Tree) and var.data == 'var' and (len(var.children) > 0) and isinstance(var.children[0], str):
            return _get_template_name(var.children[0])
        return None

    def _get_tree(self, template: TreeOrCode) -> Tree[str]:
        if isinstance(template, str):
            assert self._parse
            template = self._parse(template)
        if not isinstance(template, Tree):
            raise TypeError('template parser must return a Tree instance')
        return template

    def __call__(self, template: Tree[str]) -> 'Template':
        return Template(template, conf=self)

    def _match_tree_template(self, template: TreeOrCode, tree: Branch) -> Optional[MatchResult]:
        """Returns dict of {var: match} if found a match, else None
        """
        template_var = self.test_var(template)
        if template_var:
            if not isinstance(tree, Tree):
                raise TypeError(f'Template variables can only match Tree instances. Not {tree!r}')
            return {template_var: tree}
        if isinstance(template, str):
            if template == tree:
                return {}
            return None
        assert isinstance(template, Tree) and isinstance(tree, Tree), f'template={template} tree={tree}'
        if template.data == tree.data and len(template.children) == len(tree.children):
            res = {}
            for t1, t2 in zip(template.children, tree.children):
                matches = self._match_tree_template(t1, t2)
                if matches is None:
                    return None
                res.update(matches)
            return res
        return None