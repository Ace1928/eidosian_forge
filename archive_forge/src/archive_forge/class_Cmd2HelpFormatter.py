from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
class Cmd2HelpFormatter(argparse.RawTextHelpFormatter):
    """Custom help formatter to configure ordering of help text"""

    def _format_usage(self, usage: Optional[str], actions: Iterable[argparse.Action], groups: Iterable[argparse._ArgumentGroup], prefix: Optional[str]=None) -> str:
        if prefix is None:
            prefix = gettext('Usage: ')
        if usage is not None:
            usage %= dict(prog=self._prog)
        elif not actions:
            usage = '%(prog)s' % dict(prog=self._prog)
        else:
            prog = '%(prog)s' % dict(prog=self._prog)
            optionals = []
            positionals = []
            required_options = []
            for action in actions:
                if action.option_strings:
                    if action.required:
                        required_options.append(action)
                    else:
                        optionals.append(action)
                else:
                    positionals.append(action)
            format = self._format_actions_usage
            action_usage = format(required_options + optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:
                part_regexp = '\\(.*?\\)+|\\[.*?\\]+|\\S+'
                req_usage = format(required_options, groups)
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                req_parts = re.findall(part_regexp, req_usage)
                opt_parts = re.findall(part_regexp, opt_usage)
                pos_parts = re.findall(part_regexp, pos_usage)
                assert ' '.join(req_parts) == req_usage
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                def get_lines(parts: List[str], indent: str, prefix: Optional[str]=None) -> List[str]:
                    lines: List[str] = []
                    line: List[str] = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if req_parts:
                        lines = get_lines([prog] + req_parts, indent, prefix)
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    elif opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]
                else:
                    indent = ' ' * len(prefix)
                    parts = req_parts + opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(req_parts, indent))
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines
                usage = '\n'.join(lines)
        return '%s%s\n\n' % (prefix, usage)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts: List[str] = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
                return ', '.join(parts)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                return ', '.join(action.option_strings) + ' ' + args_string

    def _determine_metavar(self, action: argparse.Action, default_metavar: Union[str, Tuple[str, ...]]) -> Union[str, Tuple[str, ...]]:
        """Custom method to determine what to use as the metavar value of an action"""
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = [str(choice) for choice in action.choices]
            result = '{%s}' % ', '.join(choice_strs)
        else:
            result = default_metavar
        return result

    def _metavar_formatter(self, action: argparse.Action, default_metavar: Union[str, Tuple[str, ...]]) -> Callable[[int], Tuple[str, ...]]:
        metavar = self._determine_metavar(action, default_metavar)

        def format(tuple_size: int) -> Tuple[str, ...]:
            if isinstance(metavar, tuple):
                return metavar
            else:
                return (metavar,) * tuple_size
        return format

    def _format_args(self, action: argparse.Action, default_metavar: Union[str, Tuple[str, ...]]) -> str:
        """Customized to handle ranged nargs and make other output less verbose"""
        metavar = self._determine_metavar(action, default_metavar)
        metavar_formatter = self._metavar_formatter(action, default_metavar)
        nargs_range = action.get_nargs_range()
        if nargs_range is not None:
            if nargs_range[1] == constants.INFINITY:
                range_str = f'{nargs_range[0]}+'
            else:
                range_str = f'{nargs_range[0]}..{nargs_range[1]}'
            return '{}{{{}}}'.format('%s' % metavar_formatter(1), range_str)
        elif isinstance(metavar, str):
            if action.nargs == ZERO_OR_MORE:
                return '[%s [...]]' % metavar_formatter(1)
            elif action.nargs == ONE_OR_MORE:
                return '%s [...]' % metavar_formatter(1)
            elif isinstance(action.nargs, int) and action.nargs > 1:
                return '{}{{{}}}'.format('%s' % metavar_formatter(1), action.nargs)
        return super()._format_args(action, default_metavar)