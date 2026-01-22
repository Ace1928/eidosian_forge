from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
class DirectiveResolvers:

    @staticmethod
    def name(directive, _info):
        return directive.name

    @staticmethod
    def description(directive, _info):
        return directive.description

    @staticmethod
    def is_repeatable(directive, _info):
        return directive.is_repeatable

    @staticmethod
    def locations(directive, _info):
        return directive.locations

    @staticmethod
    def args(directive, _info, includeDeprecated=False):
        items = directive.args.items()
        return list(items) if includeDeprecated else [item for item in items if item[1].deprecation_reason is None]