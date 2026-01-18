from __future__ import annotations
import copy
import os
import typing as T
from .. import compilers, environment, mesonlib, optinterpreter
from .. import coredata as cdata
from ..build import Executable, Jar, SharedLibrary, SharedModule, StaticLibrary
from ..compilers import detect_compiler_for
from ..interpreterbase import InvalidArguments, SubProject
from ..mesonlib import MachineChoice, OptionKey
from ..mparser import BaseNode, ArithmeticNode, ArrayNode, ElementaryNode, IdNode, FunctionNode, BaseStringNode
from .interpreter import AstInterpreter
def traverse_nodes(inqueue: T.List[BaseNode]) -> T.List[BaseNode]:
    res: T.List[BaseNode] = []
    while inqueue:
        curr = inqueue.pop(0)
        arg_node = None
        assert isinstance(curr, BaseNode)
        if isinstance(curr, FunctionNode):
            arg_node = curr.args
        elif isinstance(curr, ArrayNode):
            arg_node = curr.args
        elif isinstance(curr, IdNode):
            assert isinstance(curr.value, str)
            var_name = curr.value
            if var_name in self.assignments:
                tmp_node = self.assignments[var_name]
                if isinstance(tmp_node, (ArrayNode, IdNode, FunctionNode)):
                    inqueue += [tmp_node]
        elif isinstance(curr, ArithmeticNode):
            inqueue += [curr.left, curr.right]
        if arg_node is None:
            continue
        arg_nodes = arg_node.arguments.copy()
        if isinstance(curr, FunctionNode) and curr.func_name.value in BUILD_TARGET_FUNCTIONS:
            arg_nodes.pop(0)
        elementary_nodes = [x for x in arg_nodes if isinstance(x, (str, BaseStringNode))]
        inqueue += [x for x in arg_nodes if isinstance(x, (FunctionNode, ArrayNode, IdNode, ArithmeticNode))]
        if elementary_nodes:
            res += [curr]
    return res