from typing import Union, Optional, Mapping, Dict, Tuple, Iterator
from lark import Tree, Transformer
from lark.exceptions import MissingVariableError
class _ReplaceVars(Transformer[str, Tree[str]]):

    def __init__(self, conf: TemplateConf, vars: Mapping[str, Tree[str]]) -> None:
        super().__init__()
        self._conf = conf
        self._vars = vars

    def __default__(self, data, children, meta) -> Tree[str]:
        tree = super().__default__(data, children, meta)
        var = self._conf.test_var(tree)
        if var:
            try:
                return self._vars[var]
            except KeyError:
                raise MissingVariableError(f'No mapping for template variable ({var})')
        return tree