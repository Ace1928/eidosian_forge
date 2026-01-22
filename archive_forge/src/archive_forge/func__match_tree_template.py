from typing import Union, Optional, Mapping, Dict, Tuple, Iterator
from lark import Tree, Transformer
from lark.exceptions import MissingVariableError
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