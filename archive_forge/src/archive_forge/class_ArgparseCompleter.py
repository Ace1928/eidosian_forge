import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
class ArgparseCompleter:
    """Automatic command line tab completion based on argparse parameters"""

    def __init__(self, parser: argparse.ArgumentParser, cmd2_app: 'Cmd', *, parent_tokens: Optional[Dict[str, List[str]]]=None) -> None:
        """
        Create an ArgparseCompleter

        :param parser: ArgumentParser instance
        :param cmd2_app: reference to the Cmd2 application that owns this ArgparseCompleter
        :param parent_tokens: optional dictionary mapping parent parsers' arg names to their tokens
                              This is only used by ArgparseCompleter when recursing on subcommand parsers
                              Defaults to None
        """
        self._parser = parser
        self._cmd2_app = cmd2_app
        if parent_tokens is None:
            parent_tokens = dict()
        self._parent_tokens = parent_tokens
        self._flags = []
        self._flag_to_action = {}
        self._positional_actions = []
        self._subcommand_action = None
        for action in self._parser._actions:
            if action.option_strings:
                for option in action.option_strings:
                    self._flags.append(option)
                    self._flag_to_action[option] = action
            else:
                self._positional_actions.append(action)
                if isinstance(action, argparse._SubParsersAction):
                    self._subcommand_action = action

    def complete(self, text: str, line: str, begidx: int, endidx: int, tokens: List[str], *, cmd_set: Optional[CommandSet]=None) -> List[str]:
        """
        Complete text using argparse metadata

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param tokens: list of argument tokens being passed to the parser
        :param cmd_set: if tab completing a command, the CommandSet the command's function belongs to, if applicable.
                        Defaults to None.

        :raises: CompletionError for various types of tab completion errors
        """
        if not tokens:
            return []
        remaining_positionals = deque(self._positional_actions)
        skip_remaining_flags = False
        pos_arg_state: Optional[_ArgumentState] = None
        flag_arg_state: Optional[_ArgumentState] = None
        matched_flags: List[str] = []
        consumed_arg_values: Dict[str, List[str]] = dict()
        completed_mutex_groups: Dict[argparse._MutuallyExclusiveGroup, argparse.Action] = dict()

        def consume_argument(arg_state: _ArgumentState) -> None:
            """Consuming token as an argument"""
            arg_state.count += 1
            consumed_arg_values.setdefault(arg_state.action.dest, [])
            consumed_arg_values[arg_state.action.dest].append(token)

        def update_mutex_groups(arg_action: argparse.Action) -> None:
            """
            Check if an argument belongs to a mutually exclusive group and either mark that group
            as complete or print an error if the group has already been completed
            :param arg_action: the action of the argument
            :raises: CompletionError if the group is already completed
            """
            for group in self._parser._mutually_exclusive_groups:
                if arg_action in group._group_actions:
                    if group in completed_mutex_groups:
                        completer_action = completed_mutex_groups[group]
                        if arg_action == completer_action:
                            return
                        error = 'Error: argument {}: not allowed with argument {}'.format(argparse._get_action_name(arg_action), argparse._get_action_name(completer_action))
                        raise CompletionError(error)
                    completed_mutex_groups[group] = arg_action
                    for group_action in group._group_actions:
                        if group_action == arg_action:
                            continue
                        elif group_action in self._flag_to_action.values():
                            matched_flags.extend(group_action.option_strings)
                        elif group_action in remaining_positionals:
                            remaining_positionals.remove(group_action)
                    break
        for token_index, token in enumerate(tokens[:-1]):
            if pos_arg_state is not None and pos_arg_state.is_remainder:
                consume_argument(pos_arg_state)
                continue
            elif flag_arg_state is not None and flag_arg_state.is_remainder:
                if token == '--':
                    flag_arg_state = None
                else:
                    consume_argument(flag_arg_state)
                continue
            elif token == '--' and (not skip_remaining_flags):
                if flag_arg_state is not None and isinstance(flag_arg_state.min, int) and (flag_arg_state.count < flag_arg_state.min):
                    raise _UnfinishedFlagError(flag_arg_state)
                else:
                    flag_arg_state = None
                    skip_remaining_flags = True
                    continue
            if _looks_like_flag(token, self._parser) and (not skip_remaining_flags):
                if flag_arg_state is not None and isinstance(flag_arg_state.min, int) and (flag_arg_state.count < flag_arg_state.min):
                    raise _UnfinishedFlagError(flag_arg_state)
                flag_arg_state = None
                action = None
                if token in self._flag_to_action:
                    action = self._flag_to_action[token]
                elif self._parser.allow_abbrev:
                    candidates_flags = [flag for flag in self._flag_to_action if flag.startswith(token)]
                    if len(candidates_flags) == 1:
                        action = self._flag_to_action[candidates_flags[0]]
                if action is not None:
                    update_mutex_groups(action)
                    if isinstance(action, (argparse._AppendAction, argparse._AppendConstAction, argparse._CountAction)):
                        consumed_arg_values.setdefault(action.dest, [])
                    else:
                        matched_flags.extend(action.option_strings)
                        consumed_arg_values[action.dest] = []
                    new_arg_state = _ArgumentState(action)
                    if new_arg_state.max > 0:
                        flag_arg_state = new_arg_state
                        skip_remaining_flags = flag_arg_state.is_remainder
            elif flag_arg_state is not None:
                consume_argument(flag_arg_state)
                if isinstance(flag_arg_state.max, (float, int)) and flag_arg_state.count >= flag_arg_state.max:
                    flag_arg_state = None
            else:
                if pos_arg_state is None:
                    if remaining_positionals:
                        action = remaining_positionals.popleft()
                        if action == self._subcommand_action:
                            if token in self._subcommand_action.choices:
                                parent_tokens = {**self._parent_tokens, **consumed_arg_values}
                                if action.dest != argparse.SUPPRESS:
                                    parent_tokens[action.dest] = [token]
                                parser: argparse.ArgumentParser = self._subcommand_action.choices[token]
                                completer_type = self._cmd2_app._determine_ap_completer_type(parser)
                                completer = completer_type(parser, self._cmd2_app, parent_tokens=parent_tokens)
                                return completer.complete(text, line, begidx, endidx, tokens[token_index + 1:], cmd_set=cmd_set)
                            else:
                                return []
                        else:
                            pos_arg_state = _ArgumentState(action)
                if pos_arg_state is not None:
                    update_mutex_groups(pos_arg_state.action)
                    consume_argument(pos_arg_state)
                    if pos_arg_state.is_remainder:
                        skip_remaining_flags = True
                    elif isinstance(pos_arg_state.max, (float, int)) and pos_arg_state.count >= pos_arg_state.max:
                        pos_arg_state = None
                        if remaining_positionals and remaining_positionals[0].nargs == argparse.REMAINDER:
                            skip_remaining_flags = True
        if _looks_like_flag(text, self._parser) and (not skip_remaining_flags):
            if flag_arg_state is not None and isinstance(flag_arg_state.min, int) and (flag_arg_state.count < flag_arg_state.min):
                raise _UnfinishedFlagError(flag_arg_state)
            return self._complete_flags(text, line, begidx, endidx, matched_flags)
        completion_results = []
        if flag_arg_state is not None:
            completion_results = self._complete_arg(text, line, begidx, endidx, flag_arg_state, consumed_arg_values, cmd_set=cmd_set)
            if completion_results:
                if not self._cmd2_app.completion_hint:
                    self._cmd2_app.completion_hint = _build_hint(self._parser, flag_arg_state.action)
                return completion_results
            elif isinstance(flag_arg_state.min, int) and flag_arg_state.count < flag_arg_state.min or not _single_prefix_char(text, self._parser) or skip_remaining_flags:
                raise _NoResultsError(self._parser, flag_arg_state.action)
        elif pos_arg_state is not None or remaining_positionals:
            if pos_arg_state is None:
                action = remaining_positionals.popleft()
                pos_arg_state = _ArgumentState(action)
            completion_results = self._complete_arg(text, line, begidx, endidx, pos_arg_state, consumed_arg_values, cmd_set=cmd_set)
            if completion_results:
                if not self._cmd2_app.completion_hint:
                    self._cmd2_app.completion_hint = _build_hint(self._parser, pos_arg_state.action)
                return completion_results
            elif not _single_prefix_char(text, self._parser) or skip_remaining_flags:
                raise _NoResultsError(self._parser, pos_arg_state.action)
        if not skip_remaining_flags and (_single_prefix_char(text, self._parser) or not remaining_positionals):
            self._cmd2_app._reset_completion_defaults()
            return self._complete_flags(text, line, begidx, endidx, matched_flags)
        return completion_results

    def _complete_flags(self, text: str, line: str, begidx: int, endidx: int, matched_flags: List[str]) -> List[str]:
        """Tab completion routine for a parsers unused flags"""
        match_against = []
        for flag in self._flags:
            if flag not in matched_flags:
                action = self._flag_to_action[flag]
                if action.help != argparse.SUPPRESS:
                    match_against.append(flag)
        matches = self._cmd2_app.basic_complete(text, line, begidx, endidx, match_against)
        matched_actions: Dict[argparse.Action, List[str]] = dict()
        for flag in matches:
            action = self._flag_to_action[flag]
            matched_actions.setdefault(action, [])
            matched_actions[action].append(flag)
        for action, option_strings in matched_actions.items():
            flag_text = ', '.join(option_strings)
            if not action.required:
                flag_text = '[' + flag_text + ']'
            self._cmd2_app.display_matches.append(flag_text)
        return matches

    def _format_completions(self, arg_state: _ArgumentState, completions: Union[List[str], List[CompletionItem]]) -> List[str]:
        """Format CompletionItems into hint table"""
        if len(completions) < 2 or not all((isinstance(c, CompletionItem) for c in completions)):
            return cast(List[str], completions)
        completion_items = cast(List[CompletionItem], completions)
        all_nums = all((isinstance(c.orig_value, numbers.Number) for c in completion_items))
        if not self._cmd2_app.matches_sorted:
            if all_nums:
                completion_items.sort(key=lambda c: c.orig_value)
            else:
                completion_items.sort(key=self._cmd2_app.default_sort_key)
            self._cmd2_app.matches_sorted = True
        if len(completions) <= self._cmd2_app.max_completion_items:
            four_spaces = 4 * ' '
            destination = arg_state.action.metavar if arg_state.action.metavar else arg_state.action.dest
            if isinstance(destination, tuple):
                tuple_index = min(len(destination) - 1, arg_state.count)
                destination = destination[tuple_index]
            desc_header = arg_state.action.get_descriptive_header()
            if desc_header is None:
                desc_header = DEFAULT_DESCRIPTIVE_HEADER
            desc_header = desc_header.replace('\t', four_spaces)
            token_width = style_aware_wcswidth(destination)
            desc_width = widest_line(desc_header)
            for item in completion_items:
                token_width = max(style_aware_wcswidth(item), token_width)
                item.description = item.description.replace('\t', four_spaces)
                desc_width = max(widest_line(item.description), desc_width)
            cols = list()
            dest_alignment = HorizontalAlignment.RIGHT if all_nums else HorizontalAlignment.LEFT
            cols.append(Column(destination.upper(), width=token_width, header_horiz_align=dest_alignment, data_horiz_align=dest_alignment))
            cols.append(Column(desc_header, width=desc_width))
            hint_table = SimpleTable(cols, divider_char=self._cmd2_app.ruler)
            table_data = [[item, item.description] for item in completion_items]
            self._cmd2_app.formatted_completions = hint_table.generate_table(table_data, row_spacing=0)
        return cast(List[str], completions)

    def complete_subcommand_help(self, text: str, line: str, begidx: int, endidx: int, tokens: List[str]) -> List[str]:
        """
        Supports cmd2's help command in the completion of subcommand names
        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param tokens: arguments passed to command/subcommand
        :return: List of subcommand completions
        """
        if self._subcommand_action is not None:
            for token_index, token in enumerate(tokens):
                if token in self._subcommand_action.choices:
                    parser: argparse.ArgumentParser = self._subcommand_action.choices[token]
                    completer_type = self._cmd2_app._determine_ap_completer_type(parser)
                    completer = completer_type(parser, self._cmd2_app)
                    return completer.complete_subcommand_help(text, line, begidx, endidx, tokens[token_index + 1:])
                elif token_index == len(tokens) - 1:
                    return self._cmd2_app.basic_complete(text, line, begidx, endidx, self._subcommand_action.choices)
                else:
                    break
        return []

    def format_help(self, tokens: List[str]) -> str:
        """
        Supports cmd2's help command in the retrieval of help text
        :param tokens: arguments passed to help command
        :return: help text of the command being queried
        """
        if self._subcommand_action is not None:
            for token_index, token in enumerate(tokens):
                if token in self._subcommand_action.choices:
                    parser: argparse.ArgumentParser = self._subcommand_action.choices[token]
                    completer_type = self._cmd2_app._determine_ap_completer_type(parser)
                    completer = completer_type(parser, self._cmd2_app)
                    return completer.format_help(tokens[token_index + 1:])
                else:
                    break
        return self._parser.format_help()

    def _complete_arg(self, text: str, line: str, begidx: int, endidx: int, arg_state: _ArgumentState, consumed_arg_values: Dict[str, List[str]], *, cmd_set: Optional[CommandSet]=None) -> List[str]:
        """
        Tab completion routine for an argparse argument
        :return: list of completions
        :raises: CompletionError if the completer or choices function this calls raises one
        """
        arg_choices: Union[List[str], ChoicesCallable]
        if arg_state.action.choices is not None:
            arg_choices = list(arg_state.action.choices)
            if not arg_choices:
                return []
            if all((isinstance(x, numbers.Number) for x in arg_choices)):
                arg_choices.sort()
                self._cmd2_app.matches_sorted = True
            for index, choice in enumerate(arg_choices):
                if not isinstance(choice, str):
                    arg_choices[index] = str(choice)
        else:
            choices_attr = arg_state.action.get_choices_callable()
            if choices_attr is None:
                return []
            arg_choices = choices_attr
        args = []
        kwargs = {}
        if isinstance(arg_choices, ChoicesCallable):
            self_arg = self._cmd2_app._resolve_func_self(arg_choices.to_call, cmd_set)
            if self_arg is None:
                raise CompletionError('Could not find CommandSet instance matching defining type for completer')
            args.append(self_arg)
            to_call_params = inspect.signature(arg_choices.to_call).parameters
            if ARG_TOKENS in to_call_params:
                arg_tokens = {**self._parent_tokens, **consumed_arg_values}
                arg_tokens.setdefault(arg_state.action.dest, [])
                arg_tokens[arg_state.action.dest].append(text)
                kwargs[ARG_TOKENS] = arg_tokens
        if isinstance(arg_choices, ChoicesCallable) and arg_choices.is_completer:
            args.extend([text, line, begidx, endidx])
            results = arg_choices.completer(*args, **kwargs)
        else:
            completion_items: List[str] = []
            if isinstance(arg_choices, ChoicesCallable):
                if not arg_choices.is_completer:
                    choices_func = arg_choices.choices_provider
                    if isinstance(choices_func, ChoicesProviderFuncWithTokens):
                        completion_items = choices_func(*args, **kwargs)
                    else:
                        completion_items = choices_func(*args)
            else:
                completion_items = arg_choices
            used_values = consumed_arg_values.get(arg_state.action.dest, [])
            completion_items = [choice for choice in completion_items if choice not in used_values]
            results = self._cmd2_app.basic_complete(text, line, begidx, endidx, completion_items)
        if not results:
            self._cmd2_app.matches_sorted = False
            return []
        return self._format_completions(arg_state, results)