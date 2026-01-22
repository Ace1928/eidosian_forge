from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _parse_declaration(first_token, tokens):
    """Parse a declaration.

    Consume :obj:`tokens` until the end of the declaration or the first error.

    :type first_token: :term:`component value`
    :param first_token: The first component value of the rule.
    :type tokens: :term:`iterator`
    :param tokens: An iterator yielding :term:`component values`.
    :returns:
        A :class:`~tinycss2.ast.Declaration`
        or :class:`~tinycss2.ast.ParseError`.

    """
    name = first_token
    if name.type != 'ident':
        return ParseError(name.source_line, name.source_column, 'invalid', 'Expected <ident> for declaration name, got %s.' % name.type)
    colon = _next_significant(tokens)
    if colon is None:
        return ParseError(name.source_line, name.source_column, 'invalid', "Expected ':' after declaration name, got EOF")
    elif colon != ':':
        return ParseError(colon.source_line, colon.source_column, 'invalid', "Expected ':' after declaration name, got %s." % colon.type)
    value = []
    state = 'value'
    for i, token in enumerate(tokens):
        if state == 'value' and token == '!':
            state = 'bang'
            bang_position = i
        elif state == 'bang' and token.type == 'ident' and (token.lower_value == 'important'):
            state = 'important'
        elif token.type not in ('whitespace', 'comment'):
            state = 'value'
        value.append(token)
    if state == 'important':
        del value[bang_position:]
    return Declaration(name.source_line, name.source_column, name.value, name.lower_value, value, state == 'important')