from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def arg_list_from_node(n):
    args = []
    if isinstance(n, FunctionNode):
        args = list(n.args.arguments)
        if n.func_name.value in BUILD_TARGET_FUNCTIONS:
            args.pop(0)
    elif isinstance(n, ArrayNode):
        args = n.args.arguments
    elif isinstance(n, ArgumentNode):
        args = n.arguments
    return args