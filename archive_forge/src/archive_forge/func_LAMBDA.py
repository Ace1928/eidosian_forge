import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def LAMBDA(self, node):
    args = []
    annotations = []
    for arg in node.args.posonlyargs:
        args.append(arg.arg)
        annotations.append(arg.annotation)
    for arg in node.args.args + node.args.kwonlyargs:
        args.append(arg.arg)
        annotations.append(arg.annotation)
    defaults = node.args.defaults + node.args.kw_defaults
    has_annotations = not isinstance(node, ast.Lambda)
    for arg_name in ('vararg', 'kwarg'):
        wildcard = getattr(node.args, arg_name)
        if not wildcard:
            continue
        args.append(wildcard.arg)
        if has_annotations:
            annotations.append(wildcard.annotation)
    if has_annotations:
        annotations.append(node.returns)
    if len(set(args)) < len(args):
        for idx, arg in enumerate(args):
            if arg in args[:idx]:
                self.report(messages.DuplicateArgument, node, arg)
    for annotation in annotations:
        self.handleAnnotation(annotation, node)
    for default in defaults:
        self.handleNode(default, node)

    def runFunction():
        with self.in_scope(FunctionScope):
            self.handleChildren(node, omit=('decorator_list', 'returns', 'type_params'))
    self.deferFunction(runFunction)