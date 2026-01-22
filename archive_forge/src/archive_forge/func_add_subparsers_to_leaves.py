from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def add_subparsers_to_leaves(root: Optional[SubparsersSpecification], leaf: SubparsersSpecification) -> SubparsersSpecification:
    if root is None:
        return leaf
    new_parsers_from_name = {}
    for name, parser in root.parser_from_name.items():
        new_parsers_from_name[name] = dataclasses.replace(parser, subparsers=add_subparsers_to_leaves(parser.subparsers, leaf))
    return dataclasses.replace(root, parser_from_name=new_parsers_from_name, required=root.required or leaf.required)