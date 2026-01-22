from collections.abc import Iterable
import decimal
from functools import partial
from wandb_graphql.language import ast
from wandb_graphql.language.printer import print_ast
from wandb_graphql.type import (GraphQLField, GraphQLList,
from .utils import to_camel_case
class DSLType(object):

    def __init__(self, type):
        self.type = type

    def __getattr__(self, name):
        formatted_name, field_def = self.get_field(name)
        return DSLField(formatted_name, field_def)

    def get_field(self, name):
        camel_cased_name = to_camel_case(name)
        if name in self.type.fields:
            return (name, self.type.fields[name])
        if camel_cased_name in self.type.fields:
            return (camel_cased_name, self.type.fields[camel_cased_name])
        raise KeyError('Field {} doesnt exist in type {}.'.format(name, self.type.name))