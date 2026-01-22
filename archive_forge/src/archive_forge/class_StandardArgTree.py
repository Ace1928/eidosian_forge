from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class StandardArgTree(ArgGroupNode):
    """Argument tree for most cmake-statement commands. Generically arguments
     are composed of a positional argument list, followed by one or more
     keyword arguments, followed by one or more flags::

      command_name(parg1 parg2 parg3...
              KEYWORD1 kwarg1 kwarg2...
              KEYWORD2 kwarg3 kwarg4...
              FLAG1 FLAG2 FLAG3)

  """

    def __init__(self):
        super(StandardArgTree, self).__init__()
        self.parg_groups = []
        self.kwarg_groups = []
        self.cmdspec = None

    def check_required_kwargs(self, lint_ctx, required_kwargs):
        for kwargnode in self.kwarg_groups:
            required_kwargs.pop(get_normalized_kwarg(kwargnode.keyword.token), None)
        if required_kwargs:
            location = self.get_location()
            for token in self.get_semantic_tokens():
                location = token.get_location()
                break
            missing_kwargs = sorted(((lintid, word) for word, lintid in sorted(required_kwargs.items())))
            for lintid, word in missing_kwargs:
                lint_ctx.record_lint(lintid, word, location=location)

    @classmethod
    def parse2(cls, ctx, tokens, cmdspec, kwargs, breakstack):
        """
    Standard parser for the commands in the form of::

        command_name(parg1 parg2 parg3...
                    KEYWORD1 kwarg1 kwarg2...
                    KEYWORD2 kwarg3 kwarg4...
                    FLAG1 FLAG2 FLAG3)
    The parser starts off as a positional parser. If a keyword or flag is
    encountered the positional parser is popped off the parse stack. If it was
    a keyword then the keyword parser is pushed on the parse stack. If it was
    a flag than a new flag parser is pushed onto the stack.
    """
        pargspecs = list(cmdspec.pargs)
        tree = cls()
        tree.cmdspec = cmdspec
        while tokens and tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        default_spec = DEFAULT_PSPEC
        if len(pargspecs) == 1 and pargspecs[0].legacy and (not npargs_is_exact(pargspecs[0].nargs)):
            default_spec = pargspecs.pop(0)
        all_flags = list(default_spec.flags)
        for pspec in pargspecs:
            all_flags.extend(pspec.flags)
        kwarg_breakstack = breakstack + [KwargBreaker(list(kwargs.keys()) + all_flags)]
        while tokens:
            if tokens[0].type in WHITESPACE_TOKENS:
                tree.children.append(tokens.pop(0))
                continue
            if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
                if comment_belongs_up_tree(ctx, tokens, tree, breakstack):
                    break
                tree.children.append(CommentNode.consume(ctx, tokens))
                continue
            if tokens[0].type in (lex.TokenType.FORMAT_OFF, lex.TokenType.FORMAT_ON):
                tree.children.append(OnOffNode.consume(ctx, tokens))
                continue
            if should_break(tokens[0], breakstack):
                if pargspecs:
                    pspec = pargspecs[0]
                else:
                    pspec = default_spec
                if not npargs_is_exact(pspec.nargs) or pspec.nargs == 0:
                    break
            ntokens = len(tokens)
            word = get_normalized_kwarg(tokens[0])
            if word in kwargs:
                with ctx.pusharg(tree):
                    subtree = KeywordGroupNode.parse(ctx, tokens, word, kwargs[word], kwarg_breakstack)
                tree.kwarg_groups.append(subtree)
            else:
                if pargspecs:
                    pspec = pargspecs.pop(0)
                else:
                    pspec = default_spec
                other_flags = []
                for otherspec in pargspecs:
                    for flag in otherspec.flags:
                        if flag in pspec.flags:
                            continue
                        other_flags.append(flag)
                positional_breakstack = breakstack + [KwargBreaker(list(kwargs.keys()) + other_flags)]
                with ctx.pusharg(tree):
                    subtree = PositionalGroupNode.parse2(ctx, tokens, pspec, positional_breakstack)
                    tree.parg_groups.append(subtree)
            if len(tokens) >= ntokens:
                raise InternalError('parsed an empty subtree at {}:\n  {}\n pspec: {}'.format(tokens[0], dump_tree_tostr([tree]), pspec))
            tree.children.append(subtree)
        return tree

    @classmethod
    def parse(cls, ctx, tokens, npargs, kwargs, flags, breakstack):
        """
    Standard parser for the commands in the form of::

        command_name(parg1 parg2 parg3...
                    KEYWORD1 kwarg1 kwarg2...
                    KEYWORD2 kwarg3 kwarg4...
                    FLAG1 FLAG2 FLAG3)

    The parser starts off as a positional parser. If a keyword or flag is
    encountered the positional parser is popped off the parse stack. If it was
    a keyword then the keyword parser is pushed on the parse stack. If it was
    a flag than a new flag parser is pushed onto the stack.
    """
        if isinstance(npargs, IMPLICIT_PARG_TYPES):
            pargspecs = [PositionalSpec(npargs, flags=flags, legacy=True)]
        else:
            assert isinstance(npargs, (list, tuple)), 'Invalid positional group specification of type {}'.format(type(npargs).__name__)
            assert flags is None, "Invalid usage of old-style 'flags' parameter with new style positional group specifications"
            pargspecs = npargs
        return cls.parse2(ctx, tokens, CommandSpec('<none>', pargspecs), kwargs, breakstack)