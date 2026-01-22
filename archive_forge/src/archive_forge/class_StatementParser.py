import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
class StatementParser:
    """Parse user input as a string into discrete command components."""

    def __init__(self, terminators: Optional[Iterable[str]]=None, multiline_commands: Optional[Iterable[str]]=None, aliases: Optional[Dict[str, str]]=None, shortcuts: Optional[Dict[str, str]]=None) -> None:
        """Initialize an instance of StatementParser.

        The following will get converted to an immutable tuple before storing internally:
        terminators, multiline commands, and shortcuts.

        :param terminators: iterable containing strings which should terminate commands
        :param multiline_commands: iterable containing the names of commands that accept multiline input
        :param aliases: dictionary containing aliases
        :param shortcuts: dictionary containing shortcuts
        """
        self.terminators: Tuple[str, ...]
        if terminators is None:
            self.terminators = (constants.MULTILINE_TERMINATOR,)
        else:
            self.terminators = tuple(terminators)
        self.multiline_commands: Tuple[str, ...] = tuple(multiline_commands) if multiline_commands is not None else ()
        self.aliases: Dict[str, str] = aliases if aliases is not None else {}
        if shortcuts is None:
            shortcuts = constants.DEFAULT_SHORTCUTS
        self.shortcuts = tuple(sorted(shortcuts.items(), key=lambda x: len(x[0]), reverse=True))
        invalid_command_chars = []
        invalid_command_chars.extend(constants.QUOTES)
        invalid_command_chars.extend(constants.REDIRECTION_CHARS)
        invalid_command_chars.extend(self.terminators)
        second_group_items = [re.escape(x) for x in invalid_command_chars]
        second_group_items.extend(['\\s', '\\Z'])
        second_group = '|'.join(second_group_items)
        expr = f'\\A\\s*(\\S*?)({second_group})'
        self._command_pattern = re.compile(expr)

    def is_valid_command(self, word: str, *, is_subcommand: bool=False) -> Tuple[bool, str]:
        """Determine whether a word is a valid name for a command.

        Commands cannot include redirection characters, whitespace,
        or termination characters. They also cannot start with a
        shortcut.

        :param word: the word to check as a command
        :param is_subcommand: Flag whether this command name is a subcommand name
        :return: a tuple of a boolean and an error string

        If word is not a valid command, return ``False`` and an error string
        suitable for inclusion in an error message of your choice::

            checkit = '>'
            valid, errmsg = statement_parser.is_valid_command(checkit)
            if not valid:
                errmsg = f"alias: {errmsg}"
        """
        valid = False
        if not isinstance(word, str):
            return (False, f'must be a string. Received {str(type(word))} instead')
        if not word:
            return (False, 'cannot be an empty string')
        if word.startswith(constants.COMMENT_CHAR):
            return (False, 'cannot start with the comment character')
        if not is_subcommand:
            for shortcut, _ in self.shortcuts:
                if word.startswith(shortcut):
                    errmsg = 'cannot start with a shortcut: '
                    errmsg += ', '.join((shortcut for shortcut, _ in self.shortcuts))
                    return (False, errmsg)
        errmsg = 'cannot contain: whitespace, quotes, '
        errchars = []
        errchars.extend(constants.REDIRECTION_CHARS)
        errchars.extend(self.terminators)
        errmsg += ', '.join([shlex.quote(x) for x in errchars])
        match = self._command_pattern.search(word)
        if match:
            if word == match.group(1):
                valid = True
                errmsg = ''
        return (valid, errmsg)

    def tokenize(self, line: str) -> List[str]:
        """
        Lex a string into a list of tokens. Shortcuts and aliases are expanded and
        comments are removed.

        :param line: the command line being lexed
        :return: A list of tokens
        :raises: Cmd2ShlexError if a shlex error occurs (e.g. No closing quotation)
        """
        line = self._expand(line)
        if line.lstrip().startswith(constants.COMMENT_CHAR):
            return []
        try:
            tokens = shlex_split(line)
        except ValueError as ex:
            raise Cmd2ShlexError(ex)
        tokens = self.split_on_punctuation(tokens)
        return tokens

    def parse(self, line: str) -> Statement:
        """
        Tokenize the input and parse it into a :class:`~cmd2.Statement` object,
        stripping comments, expanding aliases and shortcuts, and extracting output
        redirection directives.

        :param line: the command line being parsed
        :return: a new :class:`~cmd2.Statement` object
        :raises: Cmd2ShlexError if a shlex error occurs (e.g. No closing quotation)
        """
        terminator = ''
        if line[-1:] == constants.LINE_FEED:
            terminator = constants.LINE_FEED
        command = ''
        args = ''
        arg_list = []
        tokens = self.tokenize(line)
        terminator_pos = len(tokens) + 1
        for pos, cur_token in enumerate(tokens):
            for test_terminator in self.terminators:
                if cur_token.startswith(test_terminator):
                    terminator_pos = pos
                    terminator = test_terminator
                    break
            else:
                continue
            break
        if terminator:
            if terminator == constants.LINE_FEED:
                terminator_pos = len(tokens) + 1
            command, args = self._command_and_args(tokens[:terminator_pos])
            arg_list = tokens[1:terminator_pos]
            tokens = tokens[terminator_pos + 1:]
        else:
            testcommand, testargs = self._command_and_args(tokens)
            if testcommand in self.multiline_commands:
                command = testcommand
                args = testargs
                arg_list = tokens[1:]
                tokens = []
        pipe_to = ''
        output = ''
        output_to = ''
        try:
            pipe_index = tokens.index(constants.REDIRECTION_PIPE)
        except ValueError:
            pipe_index = len(tokens)
        try:
            redir_index = tokens.index(constants.REDIRECTION_OUTPUT)
        except ValueError:
            redir_index = len(tokens)
        try:
            append_index = tokens.index(constants.REDIRECTION_APPEND)
        except ValueError:
            append_index = len(tokens)
        if pipe_index < redir_index and pipe_index < append_index:
            pipe_to_tokens = tokens[pipe_index + 1:]
            utils.expand_user_in_tokens(pipe_to_tokens)
            pipe_to = ' '.join(pipe_to_tokens)
            tokens = tokens[:pipe_index]
        elif redir_index != append_index:
            if redir_index < append_index:
                output = constants.REDIRECTION_OUTPUT
                output_index = redir_index
            else:
                output = constants.REDIRECTION_APPEND
                output_index = append_index
            if len(tokens) > output_index + 1:
                unquoted_path = utils.strip_quotes(tokens[output_index + 1])
                if unquoted_path:
                    output_to = utils.expand_user(tokens[output_index + 1])
            tokens = tokens[:output_index]
        if terminator:
            suffix = ' '.join(tokens)
        else:
            suffix = ''
            if not command:
                command, args = self._command_and_args(tokens)
                arg_list = tokens[1:]
        if command in self.multiline_commands:
            multiline_command = command
        else:
            multiline_command = ''
        statement = Statement(args, raw=line, command=command, arg_list=arg_list, multiline_command=multiline_command, terminator=terminator, suffix=suffix, pipe_to=pipe_to, output=output, output_to=output_to)
        return statement

    def parse_command_only(self, rawinput: str) -> Statement:
        """Partially parse input into a :class:`~cmd2.Statement` object.

        The command is identified, and shortcuts and aliases are expanded.
        Multiline commands are identified, but terminators and output
        redirection are not parsed.

        This method is used by tab completion code and therefore must not
        generate an exception if there are unclosed quotes.

        The :class:`~cmd2.Statement` object returned by this method can at most
        contain values in the following attributes:
        :attr:`~cmd2.Statement.args`, :attr:`~cmd2.Statement.raw`,
        :attr:`~cmd2.Statement.command`,
        :attr:`~cmd2.Statement.multiline_command`

        :attr:`~cmd2.Statement.args` will include all output redirection
        clauses and command terminators.

        Different from :meth:`~cmd2.parsing.StatementParser.parse` this method
        does not remove redundant whitespace within args. However, it does
        ensure args has no leading or trailing whitespace.

        :param rawinput: the command line as entered by the user
        :return: a new :class:`~cmd2.Statement` object
        """
        line = self._expand(rawinput)
        command = ''
        args = ''
        match = self._command_pattern.search(line)
        if match:
            command = match.group(1)
            args = line[match.end(1):].strip()
            if not command or not args:
                args = ''
        if command in self.multiline_commands:
            multiline_command = command
        else:
            multiline_command = ''
        statement = Statement(args, raw=rawinput, command=command, multiline_command=multiline_command)
        return statement

    def get_command_arg_list(self, command_name: str, to_parse: Union[Statement, str], preserve_quotes: bool) -> Tuple[Statement, List[str]]:
        """
        Convenience method used by the argument parsing decorators.

        Retrieves just the arguments being passed to their ``do_*`` methods as a list.

        :param command_name: name of the command being run
        :param to_parse: what is being passed to the ``do_*`` method. It can be one of two types:

                             1. An already parsed :class:`~cmd2.Statement`
                             2. An argument string in cases where a ``do_*`` method is
                                explicitly called. Calling ``do_help('alias create')`` would
                                cause ``to_parse`` to be 'alias create'.

                                In this case, the string will be converted to a
                                :class:`~cmd2.Statement` and returned along with
                                the argument list.

        :param preserve_quotes: if ``True``, then quotes will not be stripped from
                                the arguments
        :return: A tuple containing the :class:`~cmd2.Statement` and a list of
                 strings representing the arguments
        """
        if not isinstance(to_parse, Statement):
            to_parse = self.parse(command_name + ' ' + to_parse)
        if preserve_quotes:
            return (to_parse, to_parse.arg_list)
        else:
            return (to_parse, to_parse.argv[1:])

    def _expand(self, line: str) -> str:
        """Expand aliases and shortcuts"""
        remaining_aliases = list(self.aliases.keys())
        keep_expanding = bool(remaining_aliases)
        while keep_expanding:
            keep_expanding = False
            match = self._command_pattern.search(line)
            if match:
                command = match.group(1)
                if command in remaining_aliases:
                    line = self.aliases[command] + match.group(2) + line[match.end(2):]
                    remaining_aliases.remove(command)
                    keep_expanding = bool(remaining_aliases)
        for shortcut, expansion in self.shortcuts:
            if line.startswith(shortcut):
                shortcut_len = len(shortcut)
                if len(line) == shortcut_len or line[shortcut_len] != ' ':
                    expansion += ' '
                line = line.replace(shortcut, expansion, 1)
                break
        return line

    @staticmethod
    def _command_and_args(tokens: List[str]) -> Tuple[str, str]:
        """Given a list of tokens, return a tuple of the command
        and the args as a string.
        """
        command = ''
        args = ''
        if tokens:
            command = tokens[0]
        if len(tokens) > 1:
            args = ' '.join(tokens[1:])
        return (command, args)

    def split_on_punctuation(self, tokens: List[str]) -> List[str]:
        """Further splits tokens from a command line using punctuation characters.

        Punctuation characters are treated as word breaks when they are in
        unquoted strings. Each run of punctuation characters is treated as a
        single token.

        :param tokens: the tokens as parsed by shlex
        :return: a new list of tokens, further split using punctuation
        """
        punctuation: List[str] = []
        punctuation.extend(self.terminators)
        punctuation.extend(constants.REDIRECTION_CHARS)
        punctuated_tokens = []
        for cur_initial_token in tokens:
            if len(cur_initial_token) <= 1 or cur_initial_token[0] in constants.QUOTES:
                punctuated_tokens.append(cur_initial_token)
                continue
            cur_index = 0
            cur_char = cur_initial_token[cur_index]
            new_token = ''
            while True:
                if cur_char not in punctuation:
                    while cur_char not in punctuation:
                        new_token += cur_char
                        cur_index += 1
                        if cur_index < len(cur_initial_token):
                            cur_char = cur_initial_token[cur_index]
                        else:
                            break
                else:
                    cur_punc = cur_char
                    while cur_char == cur_punc:
                        new_token += cur_char
                        cur_index += 1
                        if cur_index < len(cur_initial_token):
                            cur_char = cur_initial_token[cur_index]
                        else:
                            break
                punctuated_tokens.append(new_token)
                new_token = ''
                if cur_index >= len(cur_initial_token):
                    break
        return punctuated_tokens