import re
import copy
import inspect
import ast
import textwrap
class ExpandVarargTransformerStarred(ExpandVarargTransformer):

    def visit_Starred(self, node):
        if node.value.id == self.starred_name:
            return [ast.Name(id=name, ctx=node.ctx) for name in self.expand_names]
        else:
            return node