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
def _eval_node_name(node_id: str, context: EvaluationContext):
    policy = EVALUATION_POLICIES[context.evaluation]
    if policy.allow_locals_access and node_id in context.locals:
        return context.locals[node_id]
    if policy.allow_globals_access and node_id in context.globals:
        return context.globals[node_id]
    if policy.allow_builtins_access and hasattr(builtins, node_id):
        return getattr(builtins, node_id)
    if not policy.allow_globals_access and (not policy.allow_locals_access):
        raise GuardRejection(f'Namespace access not allowed in {context.evaluation} mode')
    else:
        raise NameError(f'{node_id} not found in locals, globals, nor builtins')