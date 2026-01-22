from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
class AstArcAnalyzer:
    """Analyze source text with an AST to find executable code paths."""

    def __init__(self, text: str, statements: set[TLineNo], multiline: dict[TLineNo, TLineNo]) -> None:
        self.root_node = ast.parse(text)
        self.statements = {multiline.get(l, l) for l in statements}
        self.multiline = multiline
        dump_ast = bool(int(os.getenv('COVERAGE_AST_DUMP', '0')))
        if dump_ast:
            print(f'Statements: {self.statements}')
            print(f'Multiline map: {self.multiline}')
            dumpkw: dict[str, Any] = {}
            if sys.version_info >= (3, 9):
                dumpkw['indent'] = 4
            print(ast.dump(self.root_node, include_attributes=True, **dumpkw))
        self.arcs: set[TArc] = set()
        self.missing_arc_fragments: TArcFragments = collections.defaultdict(list)
        self.block_stack: list[Block] = []
        self.debug = bool(int(os.getenv('COVERAGE_TRACK_ARCS', '0')))

    def analyze(self) -> None:
        """Examine the AST tree from `root_node` to determine possible arcs.

        This sets the `arcs` attribute to be a set of (from, to) line number
        pairs.

        """
        for node in ast.walk(self.root_node):
            node_name = node.__class__.__name__
            code_object_handler = getattr(self, '_code_object__' + node_name, None)
            if code_object_handler is not None:
                code_object_handler(node)

    def add_arc(self, start: TLineNo, end: TLineNo, smsg: str | None=None, emsg: str | None=None) -> None:
        """Add an arc, including message fragments to use if it is missing."""
        if self.debug:
            print(f'\nAdding possible arc: ({start}, {end}): {smsg!r}, {emsg!r}')
            print(short_stack())
        self.arcs.add((start, end))
        if smsg is not None or emsg is not None:
            self.missing_arc_fragments[start, end].append((smsg, emsg))

    def nearest_blocks(self) -> Iterable[Block]:
        """Yield the blocks in nearest-to-farthest order."""
        return reversed(self.block_stack)

    def line_for_node(self, node: ast.AST) -> TLineNo:
        """What is the right line number to use for this node?

        This dispatches to _line__Node functions where needed.

        """
        node_name = node.__class__.__name__
        handler = cast(Optional[Callable[[ast.AST], TLineNo]], getattr(self, '_line__' + node_name, None))
        if handler is not None:
            return handler(node)
        else:
            return node.lineno

    def _line_decorated(self, node: ast.FunctionDef) -> TLineNo:
        """Compute first line number for things that can be decorated (classes and functions)."""
        if node.decorator_list:
            lineno = node.decorator_list[0].lineno
        else:
            lineno = node.lineno
        return lineno

    def _line__Assign(self, node: ast.Assign) -> TLineNo:
        return self.line_for_node(node.value)
    _line__ClassDef = _line_decorated

    def _line__Dict(self, node: ast.Dict) -> TLineNo:
        if node.keys:
            if node.keys[0] is not None:
                return node.keys[0].lineno
            else:
                return node.values[0].lineno
        else:
            return node.lineno
    _line__FunctionDef = _line_decorated
    _line__AsyncFunctionDef = _line_decorated

    def _line__List(self, node: ast.List) -> TLineNo:
        if node.elts:
            return self.line_for_node(node.elts[0])
        else:
            return node.lineno

    def _line__Module(self, node: ast.Module) -> TLineNo:
        if env.PYBEHAVIOR.module_firstline_1:
            return 1
        elif node.body:
            return self.line_for_node(node.body[0])
        else:
            return 1
    OK_TO_DEFAULT = {'AnnAssign', 'Assign', 'Assert', 'AugAssign', 'Delete', 'Expr', 'Global', 'Import', 'ImportFrom', 'Nonlocal', 'Pass'}

    def add_arcs(self, node: ast.AST) -> set[ArcStart]:
        """Add the arcs for `node`.

        Return a set of ArcStarts, exits from this node to the next. Because a
        node represents an entire sub-tree (including its children), the exits
        from a node can be arbitrarily complex::

            if something(1):
                if other(2):
                    doit(3)
                else:
                    doit(5)

        There are two exits from line 1: they start at line 3 and line 5.

        """
        node_name = node.__class__.__name__
        handler = cast(Optional[Callable[[ast.AST], Set[ArcStart]]], getattr(self, '_handle__' + node_name, None))
        if handler is not None:
            return handler(node)
        else:
            if env.TESTING:
                if node_name not in self.OK_TO_DEFAULT:
                    raise RuntimeError(f'*** Unhandled: {node}')
            return {ArcStart(self.line_for_node(node))}

    def add_body_arcs(self, body: Sequence[ast.AST], from_start: ArcStart | None=None, prev_starts: set[ArcStart] | None=None) -> set[ArcStart]:
        """Add arcs for the body of a compound statement.

        `body` is the body node.  `from_start` is a single `ArcStart` that can
        be the previous line in flow before this body.  `prev_starts` is a set
        of ArcStarts that can be the previous line.  Only one of them should be
        given.

        Returns a set of ArcStarts, the exits from this body.

        """
        if prev_starts is None:
            assert from_start is not None
            prev_starts = {from_start}
        for body_node in body:
            lineno = self.line_for_node(body_node)
            first_line = self.multiline.get(lineno, lineno)
            if first_line not in self.statements:
                maybe_body_node = self.find_non_missing_node(body_node)
                if maybe_body_node is None:
                    continue
                body_node = maybe_body_node
                lineno = self.line_for_node(body_node)
            for prev_start in prev_starts:
                self.add_arc(prev_start.lineno, lineno, prev_start.cause)
            prev_starts = self.add_arcs(body_node)
        return prev_starts

    def find_non_missing_node(self, node: ast.AST) -> ast.AST | None:
        """Search `node` looking for a child that has not been optimized away.

        This might return the node you started with, or it will work recursively
        to find a child node in self.statements.

        Returns a node, or None if none of the node remains.

        """
        lineno = self.line_for_node(node)
        first_line = self.multiline.get(lineno, lineno)
        if first_line in self.statements:
            return node
        missing_fn = cast(Optional[Callable[[ast.AST], Optional[ast.AST]]], getattr(self, '_missing__' + node.__class__.__name__, None))
        if missing_fn is not None:
            ret_node = missing_fn(node)
        else:
            ret_node = None
        return ret_node

    def _missing__If(self, node: ast.If) -> ast.AST | None:
        non_missing = self.find_non_missing_node(NodeList(node.body))
        if non_missing:
            return non_missing
        if node.orelse:
            return self.find_non_missing_node(NodeList(node.orelse))
        return None

    def _missing__NodeList(self, node: NodeList) -> ast.AST | None:
        non_missing_children = []
        for child in node.body:
            maybe_child = self.find_non_missing_node(child)
            if maybe_child is not None:
                non_missing_children.append(maybe_child)
        if not non_missing_children:
            return None
        if len(non_missing_children) == 1:
            return non_missing_children[0]
        return NodeList(non_missing_children)

    def _missing__While(self, node: ast.While) -> ast.AST | None:
        body_nodes = self.find_non_missing_node(NodeList(node.body))
        if not body_nodes:
            return None
        new_while = ast.While()
        new_while.lineno = body_nodes.lineno
        new_while.test = ast.Name()
        new_while.test.lineno = body_nodes.lineno
        new_while.test.id = 'True'
        assert hasattr(body_nodes, 'body')
        new_while.body = body_nodes.body
        new_while.orelse = []
        return new_while

    def is_constant_expr(self, node: ast.AST) -> str | None:
        """Is this a compile-time constant?"""
        node_name = node.__class__.__name__
        if node_name in ['Constant', 'NameConstant', 'Num']:
            return 'Num'
        elif isinstance(node, ast.Name):
            if node.id in ['True', 'False', 'None', '__debug__']:
                return 'Name'
        return None

    def process_break_exits(self, exits: set[ArcStart]) -> None:
        """Add arcs due to jumps from `exits` being breaks."""
        for block in self.nearest_blocks():
            if block.process_break_exits(exits, self.add_arc):
                break

    def process_continue_exits(self, exits: set[ArcStart]) -> None:
        """Add arcs due to jumps from `exits` being continues."""
        for block in self.nearest_blocks():
            if block.process_continue_exits(exits, self.add_arc):
                break

    def process_raise_exits(self, exits: set[ArcStart]) -> None:
        """Add arcs due to jumps from `exits` being raises."""
        for block in self.nearest_blocks():
            if block.process_raise_exits(exits, self.add_arc):
                break

    def process_return_exits(self, exits: set[ArcStart]) -> None:
        """Add arcs due to jumps from `exits` being returns."""
        for block in self.nearest_blocks():
            if block.process_return_exits(exits, self.add_arc):
                break

    def _handle__Break(self, node: ast.Break) -> set[ArcStart]:
        here = self.line_for_node(node)
        break_start = ArcStart(here, cause="the break on line {lineno} wasn't executed")
        self.process_break_exits({break_start})
        return set()

    def _handle_decorated(self, node: ast.FunctionDef) -> set[ArcStart]:
        """Add arcs for things that can be decorated (classes and functions)."""
        main_line: TLineNo = node.lineno
        last: TLineNo | None = node.lineno
        decs = node.decorator_list
        if decs:
            last = None
            for dec_node in decs:
                dec_start = self.line_for_node(dec_node)
                if last is not None and dec_start != last:
                    self.add_arc(last, dec_start)
                last = dec_start
            assert last is not None
            self.add_arc(last, main_line)
            last = main_line
            if env.PYBEHAVIOR.trace_decorator_line_again:
                for top, bot in zip(decs, decs[1:]):
                    self.add_arc(self.line_for_node(bot), self.line_for_node(top))
                self.add_arc(self.line_for_node(decs[0]), main_line)
                self.add_arc(main_line, self.line_for_node(decs[-1]))
            if node.body:
                body_start = self.line_for_node(node.body[0])
                body_start = self.multiline.get(body_start, body_start)
        assert last is not None
        return {ArcStart(last)}
    _handle__ClassDef = _handle_decorated

    def _handle__Continue(self, node: ast.Continue) -> set[ArcStart]:
        here = self.line_for_node(node)
        continue_start = ArcStart(here, cause="the continue on line {lineno} wasn't executed")
        self.process_continue_exits({continue_start})
        return set()

    def _handle__For(self, node: ast.For) -> set[ArcStart]:
        start = self.line_for_node(node.iter)
        self.block_stack.append(LoopBlock(start=start))
        from_start = ArcStart(start, cause='the loop on line {lineno} never started')
        exits = self.add_body_arcs(node.body, from_start=from_start)
        for xit in exits:
            self.add_arc(xit.lineno, start, xit.cause)
        my_block = self.block_stack.pop()
        assert isinstance(my_block, LoopBlock)
        exits = my_block.break_exits
        from_start = ArcStart(start, cause="the loop on line {lineno} didn't complete")
        if node.orelse:
            else_exits = self.add_body_arcs(node.orelse, from_start=from_start)
            exits |= else_exits
        else:
            exits.add(from_start)
        return exits
    _handle__AsyncFor = _handle__For
    _handle__FunctionDef = _handle_decorated
    _handle__AsyncFunctionDef = _handle_decorated

    def _handle__If(self, node: ast.If) -> set[ArcStart]:
        start = self.line_for_node(node.test)
        from_start = ArcStart(start, cause='the condition on line {lineno} was never true')
        exits = self.add_body_arcs(node.body, from_start=from_start)
        from_start = ArcStart(start, cause='the condition on line {lineno} was never false')
        exits |= self.add_body_arcs(node.orelse, from_start=from_start)
        return exits
    if sys.version_info >= (3, 10):

        def _handle__Match(self, node: ast.Match) -> set[ArcStart]:
            start = self.line_for_node(node)
            last_start = start
            exits = set()
            had_wildcard = False
            for case in node.cases:
                case_start = self.line_for_node(case.pattern)
                pattern = case.pattern
                while isinstance(pattern, ast.MatchOr):
                    pattern = pattern.patterns[-1]
                if isinstance(pattern, ast.MatchAs):
                    had_wildcard = True
                self.add_arc(last_start, case_start, 'the pattern on line {lineno} always matched')
                from_start = ArcStart(case_start, cause='the pattern on line {lineno} never matched')
                exits |= self.add_body_arcs(case.body, from_start=from_start)
                last_start = case_start
            if not had_wildcard:
                exits.add(ArcStart(case_start, cause='the pattern on line {lineno} always matched'))
            return exits

    def _handle__NodeList(self, node: NodeList) -> set[ArcStart]:
        start = self.line_for_node(node)
        exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
        return exits

    def _handle__Raise(self, node: ast.Raise) -> set[ArcStart]:
        here = self.line_for_node(node)
        raise_start = ArcStart(here, cause="the raise on line {lineno} wasn't executed")
        self.process_raise_exits({raise_start})
        return set()

    def _handle__Return(self, node: ast.Return) -> set[ArcStart]:
        here = self.line_for_node(node)
        return_start = ArcStart(here, cause="the return on line {lineno} wasn't executed")
        self.process_return_exits({return_start})
        return set()

    def _handle__Try(self, node: ast.Try) -> set[ArcStart]:
        if node.handlers:
            handler_start = self.line_for_node(node.handlers[0])
        else:
            handler_start = None
        if node.finalbody:
            final_start = self.line_for_node(node.finalbody[0])
        else:
            final_start = None
        assert handler_start is not None or final_start is not None
        try_block = TryBlock(handler_start, final_start)
        self.block_stack.append(try_block)
        start = self.line_for_node(node)
        exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
        if node.finalbody:
            try_block.handler_start = None
            if node.handlers:
                try_block.raise_from = set()
        else:
            self.block_stack.pop()
        handler_exits: set[ArcStart] = set()
        if node.handlers:
            last_handler_start: TLineNo | None = None
            for handler_node in node.handlers:
                handler_start = self.line_for_node(handler_node)
                if last_handler_start is not None:
                    self.add_arc(last_handler_start, handler_start)
                last_handler_start = handler_start
                from_cause = "the exception caught by line {lineno} didn't happen"
                from_start = ArcStart(handler_start, cause=from_cause)
                handler_exits |= self.add_body_arcs(handler_node.body, from_start=from_start)
        if node.orelse:
            exits = self.add_body_arcs(node.orelse, prev_starts=exits)
        exits |= handler_exits
        if node.finalbody:
            self.block_stack.pop()
            final_from = exits | try_block.break_from | try_block.continue_from | try_block.raise_from | try_block.return_from
            final_exits = self.add_body_arcs(node.finalbody, prev_starts=final_from)
            if try_block.break_from:
                if env.PYBEHAVIOR.finally_jumps_back:
                    for break_line in try_block.break_from:
                        lineno = break_line.lineno
                        cause = break_line.cause.format(lineno=lineno)
                        for final_exit in final_exits:
                            self.add_arc(final_exit.lineno, lineno, cause)
                    breaks = try_block.break_from
                else:
                    breaks = self._combine_finally_starts(try_block.break_from, final_exits)
                self.process_break_exits(breaks)
            if try_block.continue_from:
                if env.PYBEHAVIOR.finally_jumps_back:
                    for continue_line in try_block.continue_from:
                        lineno = continue_line.lineno
                        cause = continue_line.cause.format(lineno=lineno)
                        for final_exit in final_exits:
                            self.add_arc(final_exit.lineno, lineno, cause)
                    continues = try_block.continue_from
                else:
                    continues = self._combine_finally_starts(try_block.continue_from, final_exits)
                self.process_continue_exits(continues)
            if try_block.raise_from:
                self.process_raise_exits(self._combine_finally_starts(try_block.raise_from, final_exits))
            if try_block.return_from:
                if env.PYBEHAVIOR.finally_jumps_back:
                    for return_line in try_block.return_from:
                        lineno = return_line.lineno
                        cause = return_line.cause.format(lineno=lineno)
                        for final_exit in final_exits:
                            self.add_arc(final_exit.lineno, lineno, cause)
                    returns = try_block.return_from
                else:
                    returns = self._combine_finally_starts(try_block.return_from, final_exits)
                self.process_return_exits(returns)
            if exits:
                exits = final_exits
        return exits

    def _combine_finally_starts(self, starts: set[ArcStart], exits: set[ArcStart]) -> set[ArcStart]:
        """Helper for building the cause of `finally` branches.

        "finally" clauses might not execute their exits, and the causes could
        be due to a failure to execute any of the exits in the try block. So
        we use the causes from `starts` as the causes for `exits`.
        """
        causes = []
        for start in sorted(starts):
            if start.cause:
                causes.append(start.cause.format(lineno=start.lineno))
        cause = ' or '.join(causes)
        exits = {ArcStart(xit.lineno, cause) for xit in exits}
        return exits

    def _handle__While(self, node: ast.While) -> set[ArcStart]:
        start = to_top = self.line_for_node(node.test)
        constant_test = self.is_constant_expr(node.test)
        top_is_body0 = False
        if constant_test:
            top_is_body0 = True
        if env.PYBEHAVIOR.keep_constant_test:
            top_is_body0 = False
        if top_is_body0:
            to_top = self.line_for_node(node.body[0])
        self.block_stack.append(LoopBlock(start=to_top))
        from_start = ArcStart(start, cause='the condition on line {lineno} was never true')
        exits = self.add_body_arcs(node.body, from_start=from_start)
        for xit in exits:
            self.add_arc(xit.lineno, to_top, xit.cause)
        exits = set()
        my_block = self.block_stack.pop()
        assert isinstance(my_block, LoopBlock)
        exits.update(my_block.break_exits)
        from_start = ArcStart(start, cause='the condition on line {lineno} was never false')
        if node.orelse:
            else_exits = self.add_body_arcs(node.orelse, from_start=from_start)
            exits |= else_exits
        elif not constant_test:
            exits.add(from_start)
        return exits

    def _handle__With(self, node: ast.With) -> set[ArcStart]:
        start = self.line_for_node(node)
        if env.PYBEHAVIOR.exit_through_with:
            self.block_stack.append(WithBlock(start=start))
        exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
        if env.PYBEHAVIOR.exit_through_with:
            with_block = self.block_stack.pop()
            assert isinstance(with_block, WithBlock)
            with_exit = {ArcStart(start)}
            if exits:
                for xit in exits:
                    self.add_arc(xit.lineno, start)
                exits = with_exit
            if with_block.break_from:
                self.process_break_exits(self._combine_finally_starts(with_block.break_from, with_exit))
            if with_block.continue_from:
                self.process_continue_exits(self._combine_finally_starts(with_block.continue_from, with_exit))
            if with_block.return_from:
                self.process_return_exits(self._combine_finally_starts(with_block.return_from, with_exit))
        return exits
    _handle__AsyncWith = _handle__With

    def _code_object__Module(self, node: ast.Module) -> None:
        start = self.line_for_node(node)
        if node.body:
            exits = self.add_body_arcs(node.body, from_start=ArcStart(-start))
            for xit in exits:
                self.add_arc(xit.lineno, -start, xit.cause, "didn't exit the module")
        else:
            self.add_arc(-start, start)
            self.add_arc(start, -start)

    def _code_object__FunctionDef(self, node: ast.FunctionDef) -> None:
        start = self.line_for_node(node)
        self.block_stack.append(FunctionBlock(start=start, name=node.name))
        exits = self.add_body_arcs(node.body, from_start=ArcStart(-start))
        self.process_return_exits(exits)
        self.block_stack.pop()
    _code_object__AsyncFunctionDef = _code_object__FunctionDef

    def _code_object__ClassDef(self, node: ast.ClassDef) -> None:
        start = self.line_for_node(node)
        self.add_arc(-start, start)
        exits = self.add_body_arcs(node.body, from_start=ArcStart(start))
        for xit in exits:
            self.add_arc(xit.lineno, -start, xit.cause, f"didn't exit the body of class {node.name!r}")
    _code_object__Lambda = _make_expression_code_method('lambda')
    _code_object__GeneratorExp = _make_expression_code_method('generator expression')
    if env.PYBEHAVIOR.comprehensions_are_functions:
        _code_object__DictComp = _make_expression_code_method('dictionary comprehension')
        _code_object__SetComp = _make_expression_code_method('set comprehension')
        _code_object__ListComp = _make_expression_code_method('list comprehension')