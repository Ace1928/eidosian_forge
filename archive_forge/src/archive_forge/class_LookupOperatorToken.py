from ..helpers import iter_sequence
from ..sequence_types import is_sequence_type, match_sequence_type
from ..xpath_tokens import XPathToken, ProxyToken, XPathFunction, XPathMap, XPathArray
from .xpath31_parser import XPath31Parser
class LookupOperatorToken(XPathToken):
    """
    Question mark symbol is used for XP31+ lookup operator and also for
    placeholder in XP30+ partial functions and for optional occurrences.
    """
    symbol = lookup_name = '?'
    lbp = 85
    rbp = 85

    def __init__(self, parser, value=None):
        super().__init__(parser, value)
        if self.parser.token.symbol in ('(', ','):
            self.lbp = self.rbp = 0

    @property
    def source(self) -> str:
        if not self:
            return '?'
        elif len(self) == 1:
            return f'?{self[0].source}'
        else:
            return f'{self[0].source}?{self[1].source}'

    def nud(self):
        try:
            self.parser.expected_next('(name)', '(integer)', '(', '*')
        except SyntaxError:
            if self.lbp:
                raise
            return self
        else:
            self[:] = (self.parser.expression(85),)
            return self

    def led(self, left):
        try:
            self.parser.expected_next('(name)', '(integer)', '(', '*')
        except SyntaxError:
            if is_sequence_type(left.value, self.parser):
                self.lbp = self.rbp = 0
                left.occurrence = '?'
                return left
            raise
        else:
            self[:] = (left, self.parser.expression(85))
            return self

    def evaluate(self, context=None):
        if not self:
            return self.value
        return [x for x in self.select(context)]

    def select(self, context=None):
        if not self:
            yield from iter_sequence(self.value)
            return
        if len(self) == 1:
            if context is None:
                raise self.missing_context()
            items = (context.item,)
        else:
            items = self[0].select(context)
        for item in items:
            symbol = self[-1].symbol
            if isinstance(item, XPathMap):
                if symbol == '*':
                    for value in item.values(context):
                        yield from iter_sequence(value)
                elif symbol in ('(name)', '(integer)'):
                    yield from iter_sequence(item(self[-1].value, context=context))
                elif symbol == '(':
                    for value in self[-1].select(context):
                        yield from iter_sequence(item(self.data_value(value), context=context))
            elif isinstance(item, XPathArray):
                if symbol == '*':
                    yield from item.items(context)
                elif symbol == '(name)':
                    raise self.error('XPTY0004')
                elif symbol == '(integer)':
                    yield item(self[-1].value, context=context)
                elif symbol == '(':
                    for value in self[-1].select(context):
                        yield item(self.data_value(value), context=context)
            elif not item and isinstance(item, list):
                continue
            else:
                raise self.error('XPTY0004')