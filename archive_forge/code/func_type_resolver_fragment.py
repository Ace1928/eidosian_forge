import sys
from collections.abc import Iterable
from functools import partial
from wandb_promise import Promise, is_thenable
from ...error import GraphQLError, GraphQLLocatedError
from ...type import (GraphQLEnumType, GraphQLInterfaceType, GraphQLList,
from ..base import default_resolve_fn
from ...execution import executor
from .utils import imap, normal_map
def type_resolver_fragment(return_type, resolver, fragment, exe_context, info, catch_error):
    on_complete_type_error = partial(on_error, exe_context, info, catch_error)
    complete_object_value_resolve = partial(complete_object_value, fragment.resolve, exe_context, on_complete_type_error)
    on_resolve_error = partial(on_error, exe_context, info, catch_error)
    return partial(on_complete_resolver, on_resolve_error, complete_object_value_resolve, exe_context, info, resolver)