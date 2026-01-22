from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
@dataclasses.dataclass(frozen=True)
class ParserSpecification:
    """Each parser contains a list of arguments and optionally some subparsers."""
    f: Callable
    description: str
    args: List[_arguments.ArgumentDefinition]
    field_list: List[_fields.FieldDefinition]
    child_from_prefix: Dict[str, ParserSpecification]
    helptext_from_intern_prefixed_field_name: Dict[str, Optional[str]]
    subparsers: Optional[SubparsersSpecification]
    subparsers_from_intern_prefix: Dict[str, SubparsersSpecification]
    intern_prefix: str
    extern_prefix: str
    has_required_args: bool
    consolidate_subcommand_args: bool

    @staticmethod
    def from_callable_or_type(f: Callable[..., T], description: Optional[str], parent_classes: Set[Type[Any]], default_instance: Union[T, _fields.PropagatingMissingType, _fields.NonpropagatingMissingType], intern_prefix: str, extern_prefix: str, subcommand_prefix: str='', support_single_arg_types: bool=False) -> ParserSpecification:
        """Create a parser definition from a callable or type."""
        markers = _resolver.unwrap_annotated(f, _markers._Marker)[1]
        consolidate_subcommand_args = _markers.ConsolidateSubcommandArgs in markers
        f, type_from_typevar, field_list = _fields.field_list_from_callable(f=f, default_instance=default_instance, support_single_arg_types=support_single_arg_types)
        for i in range(len(field_list)):
            field_list[i] = dataclasses.replace(field_list[i], markers=field_list[i].markers | set(markers))
        if f in parent_classes and f is not dict:
            raise _instantiators.UnsupportedTypeAnnotationError(f'Found a cyclic dependency with type {f}.')
        parent_classes = parent_classes | {cast(Type, f)}
        has_required_args = False
        args = []
        helptext_from_intern_prefixed_field_name: Dict[str, Optional[str]] = {}
        child_from_prefix: Dict[str, ParserSpecification] = {}
        subparsers = None
        subparsers_from_prefix = {}
        for field in field_list:
            field_out = handle_field(field, type_from_typevar=type_from_typevar, parent_classes=parent_classes, intern_prefix=intern_prefix, extern_prefix=extern_prefix, subcommand_prefix=subcommand_prefix)
            if isinstance(field_out, _arguments.ArgumentDefinition):
                args.append(field_out)
                if field_out.lowered.required:
                    has_required_args = True
            elif isinstance(field_out, SubparsersSpecification):
                subparsers_from_prefix[field_out.intern_prefix] = field_out
                subparsers = add_subparsers_to_leaves(subparsers, field_out)
            elif isinstance(field_out, ParserSpecification):
                nested_parser = field_out
                child_from_prefix[field_out.intern_prefix] = nested_parser
                if nested_parser.has_required_args:
                    has_required_args = True
                if nested_parser.subparsers is not None:
                    subparsers_from_prefix.update(nested_parser.subparsers_from_intern_prefix)
                    subparsers = add_subparsers_to_leaves(subparsers, nested_parser.subparsers)
                class_field_name = _strings.make_field_name([field.intern_name])
                if field.helptext is not None:
                    helptext_from_intern_prefixed_field_name[class_field_name] = field.helptext
                else:
                    helptext_from_intern_prefixed_field_name[class_field_name] = _docstrings.get_callable_description(nested_parser.f)
                if len(nested_parser.args) >= 1 and _markers._OPTIONAL_GROUP in nested_parser.args[0].field.markers:
                    current_helptext = helptext_from_intern_prefixed_field_name[class_field_name]
                    helptext_from_intern_prefixed_field_name[class_field_name] = ('' if current_helptext is None else current_helptext + '\n\n') + 'Default: ' + str(field.default)
        return ParserSpecification(f=f, description=_strings.remove_single_line_breaks(description if description is not None else _docstrings.get_callable_description(f)), args=args, field_list=field_list, child_from_prefix=child_from_prefix, helptext_from_intern_prefixed_field_name=helptext_from_intern_prefixed_field_name, subparsers=subparsers, subparsers_from_intern_prefix=subparsers_from_prefix, intern_prefix=intern_prefix, extern_prefix=extern_prefix, has_required_args=has_required_args, consolidate_subcommand_args=consolidate_subcommand_args)

    def apply(self, parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, ...]:
        """Create defined arguments and subparsers."""
        parser.description = self.description
        subparser_group = None
        if self.subparsers is not None:
            leaves = self.subparsers.apply(parser)
            subparser_group = parser._action_groups.pop()
        else:
            leaves = (parser,)
        if self.consolidate_subcommand_args:
            for leaf in leaves:
                self.apply_args(leaf)
        else:
            self.apply_args(parser)
        if subparser_group is not None:
            parser._action_groups.append(subparser_group)
        assert parser._action_groups[1].title in ('optional arguments', 'options')
        parser._action_groups[1].title = 'options'
        return leaves

    def apply_args(self, parser: argparse.ArgumentParser, parent: Optional[ParserSpecification]=None) -> None:
        """Create defined arguments and subparsers."""

        def format_group_name(prefix: str) -> str:
            return (prefix + ' options').strip()
        group_from_prefix: Dict[str, argparse._ArgumentGroup] = {'': parser._action_groups[1], **{cast(str, group.title).partition(' ')[0]: group for group in parser._action_groups[2:]}}
        positional_group = parser._action_groups[0]
        assert positional_group.title == 'positional arguments'
        for arg in self.args:
            if arg.lowered.help is not argparse.SUPPRESS and arg.extern_prefix not in group_from_prefix:
                description = parent.helptext_from_intern_prefixed_field_name.get(arg.intern_prefix) if parent is not None else None
                group_from_prefix[arg.extern_prefix] = parser.add_argument_group(format_group_name(arg.extern_prefix), description=description)
        for arg in self.args:
            if arg.field.is_positional():
                arg.add_argument(positional_group)
                continue
            if arg.extern_prefix in group_from_prefix:
                arg.add_argument(group_from_prefix[arg.extern_prefix])
            else:
                assert arg.lowered.help is argparse.SUPPRESS
                arg.add_argument(group_from_prefix[''])
        for child in self.child_from_prefix.values():
            child.apply_args(parser, parent=self)