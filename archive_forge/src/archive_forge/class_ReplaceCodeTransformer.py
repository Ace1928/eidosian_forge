from ast import (
import ast
import copy
from typing import Dict, Optional, Union
class ReplaceCodeTransformer(NodeTransformer):
    enabled: bool = True
    debug: bool = False
    mangler: Mangler

    def __init__(self, template: Module, mapping: Optional[Dict]=None, mangling_predicate=None):
        assert isinstance(mapping, (dict, type(None)))
        assert isinstance(mangling_predicate, (type(None), type(lambda: None)))
        assert isinstance(template, ast.Module)
        self.template = template
        self.mangler = Mangler(predicate=mangling_predicate)
        if mapping is None:
            mapping = {}
        self.mapping = mapping

    @classmethod
    def from_string(cls, template: str, mapping: Optional[Dict]=None, mangling_predicate=None):
        return cls(ast.parse(template), mapping=mapping, mangling_predicate=mangling_predicate)

    def visit_Module(self, code):
        if not self.enabled:
            return code
        last = code.body[-1]
        if isinstance(last, Expr):
            code.body.pop()
            code.body.append(Assign([Name('ret-tmp', ctx=Store())], value=last.value))
            ast.fix_missing_locations(code)
            ret = Expr(value=Name('ret-tmp', ctx=Load()))
            ret = ast.fix_missing_locations(ret)
            self.mapping['__ret__'] = ret
        else:
            self.mapping['__ret__'] = ast.parse('None').body[0]
        self.mapping['__code__'] = code.body
        tpl = ast.fix_missing_locations(self.template)
        tx = copy.deepcopy(tpl)
        tx = self.mangler.visit(tx)
        node = self.generic_visit(tx)
        node_2 = ast.fix_missing_locations(node)
        if self.debug:
            print('---- Transformed code ----')
            print(ast.unparse(node_2))
            print('---- ---------------- ----')
        return node_2

    def visit_Expr(self, expr):
        if isinstance(expr.value, Name) and expr.value.id in self.mapping:
            if self.mapping[expr.value.id] is not None:
                return copy.deepcopy(self.mapping[expr.value.id])
        return self.generic_visit(expr)