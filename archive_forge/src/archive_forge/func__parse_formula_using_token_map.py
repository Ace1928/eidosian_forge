from typing import (
import cmath
import re
import numpy as np
import sympy
def _parse_formula_using_token_map(text: str, token_map: Dict[str, _HangingToken]) -> _ResolvedToken:
    """Parses a value from an infix arithmetic expression."""
    tokens: List[_HangingToken] = [_translate_token(e, token_map) for e in _tokenize(text)]
    if len(tokens) and isinstance(tokens[-1], _CustomQuirkOperationToken) and (tokens[-1].priority is not None):
        tokens = tokens[:-1]
    ops: List[Union[str, _HangingNode]] = []
    vals: List[Optional[_HangingToken]] = []

    def is_valid_end_token(tok: _HangingToken) -> bool:
        return tok != '(' and (not isinstance(tok, _CustomQuirkOperationToken))

    def is_valid_end_state() -> bool:
        return len(vals) == 1 and len(ops) == 0

    def apply(op: Union[str, _HangingNode]) -> None:
        assert isinstance(op, _HangingNode)
        if len(vals) < 2:
            raise ValueError('Bad expression: operated on nothing.\ntext={text!r}')
        b = vals.pop()
        a = vals.pop()
        vals.append(op.func(a, b))

    def close_paren() -> None:
        while True:
            if len(ops) == 0:
                raise ValueError("Bad expression: unmatched ')'.\ntext={text!r}")
            op = ops.pop()
            if op == '(':
                break
            apply(op)

    def burn_ops(w: float) -> None:
        while len(ops) and len(vals) >= 2 and (vals[-1] is not None):
            top = ops[-1]
            if not isinstance(top, _HangingNode) or top.weight is None or top.weight < w:
                break
            apply(ops.pop())

    def feed_op(could_be_binary: bool, token: Any) -> None:
        mul = cast(_CustomQuirkOperationToken, token_map['*'])
        if could_be_binary and token != ')':
            if not isinstance(token, _CustomQuirkOperationToken) or token.binary_action is None:
                burn_ops(mul.priority)
                ops.append(_HangingNode(func=cast(Callable[[T, T], T], mul.binary_action), weight=mul.priority))
        if isinstance(token, _CustomQuirkOperationToken):
            if could_be_binary and token.binary_action is not None:
                burn_ops(token.priority)
                ops.append(_HangingNode(func=token.binary_action, weight=token.priority))
            elif token.unary_action is not None:
                burn_ops(token.priority)
                vals.append(None)
                ops.append(_HangingNode(func=lambda _, b: token.unary_action(b), weight=np.inf))
            elif token.binary_action is not None:
                raise ValueError('Bad expression: binary op in bad spot.\ntext={text!r}')
    was_valid_end_token = False
    for token in tokens:
        feed_op(was_valid_end_token, token)
        was_valid_end_token = is_valid_end_token(token)
        if token == '(':
            ops.append('(')
        elif token == ')':
            close_paren()
        elif was_valid_end_token:
            vals.append(token)
    burn_ops(-np.inf)
    if not is_valid_end_state():
        raise ValueError(f'Incomplete expression.\ntext={text!r}')
    return cast(_ResolvedToken, vals[0])