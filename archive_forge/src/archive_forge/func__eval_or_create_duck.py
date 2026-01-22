from inspect import isclass, signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.decorators import undoc
def _eval_or_create_duck(duck_type, node: ast.Call, context: EvaluationContext):
    policy = EVALUATION_POLICIES[context.evaluation]
    if policy.can_call(duck_type) and (not node.keywords):
        args = [eval_node(arg, context) for arg in node.args]
        return duck_type(*args)
    return _create_duck_for_heap_type(duck_type)