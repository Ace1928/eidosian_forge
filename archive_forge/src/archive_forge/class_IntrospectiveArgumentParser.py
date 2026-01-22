from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
class IntrospectiveArgumentParser(ArgumentParser):
    """The following is a verbatim copy of ArgumentParser._parse_known_args (Python 2.7.3),
    except for the lines that contain the string "Added by argcomplete".
    """

    def _parse_known_args(self, arg_strings, namespace):
        _num_consumed_args.clear()
        self._argcomplete_namespace = namespace
        self.active_actions: List[Action] = []
        if self.fromfile_prefix_chars is not None:
            arg_strings = self._read_args_from_files(arg_strings)
        action_conflicts: Dict[Action, List[Action]] = {}
        self._action_conflicts = action_conflicts
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                conflicts = action_conflicts.setdefault(mutex_action, [])
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])
        option_string_indices = {}
        arg_string_pattern_parts = []
        arg_strings_iter = iter(arg_strings)
        for i, arg_string in enumerate(arg_strings_iter):
            if arg_string == '--':
                arg_string_pattern_parts.append('-')
                for arg_string in arg_strings_iter:
                    arg_string_pattern_parts.append('A')
            else:
                option_tuple = self._parse_optional(arg_string)
                if option_tuple is None:
                    pattern = 'A'
                else:
                    option_string_indices[i] = option_tuple
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)
        arg_strings_pattern = ''.join(arg_string_pattern_parts)
        seen_actions: Set[Action] = set()
        seen_non_default_actions: Set[Action] = set()
        self._seen_non_default_actions = seen_non_default_actions

        def take_action(action, argument_strings, option_string=None):
            seen_actions.add(action)
            argument_values = self._get_values(action, argument_strings)
            if argument_values is not action.default:
                seen_non_default_actions.add(action)
                for conflict_action in action_conflicts.get(action, []):
                    if conflict_action in seen_non_default_actions:
                        msg = gettext('not allowed with argument %s')
                        action_name = _get_action_name(conflict_action)
                        raise ArgumentError(action, msg % action_name)
            if argument_values is not SUPPRESS or isinstance(action, _SubParsersAction):
                try:
                    action(self, namespace, argument_values, option_string)
                except BaseException:
                    if isinstance(action, _SubParsersAction):
                        subnamespace = action._name_parser_map[argument_values[0]]._argcomplete_namespace
                        for key, value in vars(subnamespace).items():
                            setattr(namespace, key, value)
                    raise

        def consume_optional(start_index):
            option_tuple = option_string_indices[start_index]
            action, option_string, explicit_arg = option_tuple
            match_argument = self._match_argument
            action_tuples: List[Tuple[Action, List[str], str]] = []
            while True:
                if action is None:
                    extras.append(arg_strings[start_index])
                    return start_index + 1
                if explicit_arg is not None:
                    arg_count = match_argument(action, 'A')
                    chars = self.prefix_chars
                    if arg_count == 0 and option_string[1] not in chars:
                        action_tuples.append((action, [], option_string))
                        char = option_string[0]
                        option_string = char + explicit_arg[0]
                        new_explicit_arg = explicit_arg[1:] or None
                        optionals_map = self._option_string_actions
                        if option_string in optionals_map:
                            action = optionals_map[option_string]
                            explicit_arg = new_explicit_arg
                        else:
                            msg = gettext('ignored explicit argument %r')
                            raise ArgumentError(action, msg % explicit_arg)
                    elif arg_count == 1:
                        stop = start_index + 1
                        args = [explicit_arg]
                        action_tuples.append((action, args, option_string))
                        break
                    else:
                        msg = gettext('ignored explicit argument %r')
                        raise ArgumentError(action, msg % explicit_arg)
                else:
                    start = start_index + 1
                    selected_patterns = arg_strings_pattern[start:]
                    self.active_actions = [action]
                    _num_consumed_args[action] = 0
                    arg_count = match_argument(action, selected_patterns)
                    stop = start + arg_count
                    args = arg_strings[start:stop]
                    _num_consumed_args[action] = len(args)
                    if not action_is_open(action):
                        self.active_actions.remove(action)
                    action_tuples.append((action, args, option_string))
                    break
            assert action_tuples
            for action, args, option_string in action_tuples:
                take_action(action, args, option_string)
            return stop
        positionals = self._get_positional_actions()

        def consume_positionals(start_index):
            match_partial = self._match_arguments_partial
            selected_pattern = arg_strings_pattern[start_index:]
            arg_counts = match_partial(positionals, selected_pattern)
            for action, arg_count in zip(positionals, arg_counts):
                self.active_actions.append(action)
            for action, arg_count in zip(positionals, arg_counts):
                args = arg_strings[start_index:start_index + arg_count]
                start_index += arg_count
                _num_consumed_args[action] = len(args)
                take_action(action, args)
            positionals[:] = positionals[len(arg_counts):]
            return start_index
        extras = []
        start_index = 0
        if option_string_indices:
            max_option_string_index = max(option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:
            next_option_string_index = min([index for index in option_string_indices if index >= start_index])
            if start_index != next_option_string_index:
                positionals_end_index = consume_positionals(start_index)
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index
            if start_index not in option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                extras.extend(strings)
                start_index = next_option_string_index
            start_index = consume_optional(start_index)
        stop_index = consume_positionals(start_index)
        extras.extend(arg_strings[stop_index:])
        if positionals:
            self.active_actions.append(positionals[0])
            self.error(gettext('too few arguments'))
        for action in self._actions:
            if action.required:
                if action not in seen_actions:
                    name = _get_action_name(action)
                    self.error(gettext('argument %s is required') % name)
        for group in self._mutually_exclusive_groups:
            if group.required:
                for action in group._group_actions:
                    if action in seen_non_default_actions:
                        break
                else:
                    names = [str(_get_action_name(action)) for action in group._group_actions if action.help is not SUPPRESS]
                    msg = gettext('one of the arguments %s is required')
                    self.error(msg % ' '.join(names))
        return (namespace, extras)