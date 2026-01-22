from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
class SVGTransformationParser(object):
    """ Parse SVG transform="" data into a list of commands.

    Each distinct command will take the form of a tuple (type, data). The
    `type` is the character string that defines the type of transformation in the
    transform data, so either of "translate", "rotate", "scale", "matrix",
    "skewX" and "skewY". Data is always a list of numbers contained within the
    transformation's parentheses.

    See the SVG documentation for the interpretation of the individual elements
    for each transformation.

    The main method is `parse(text)`. It can only consume actual strings, not
    filelike objects or iterators.
    """

    def __init__(self, lexer=svg_lexer):
        self.lexer = lexer
        self.command_dispatch = {'translate': self.rule_1or2numbers, 'scale': self.rule_1or2numbers, 'skewX': self.rule_1number, 'skewY': self.rule_1number, 'rotate': self.rule_1or3numbers, 'matrix': self.rule_6numbers}
        self.number_tokens = list(['int', 'float'])

    def parse(self, text):
        """ Parse a string of SVG transform="" data.
        """
        gen = self.lexer.lex(text)
        next_val_fn = partial(next, *(gen,))
        commands = []
        token = next_val_fn()
        while token[0] is not EOF:
            command, token = self.rule_svg_transform(next_val_fn, token)
            commands.append(command)
        return commands

    def rule_svg_transform(self, next_val_fn, token):
        if token[0] != 'command':
            raise SyntaxError('expecting a transformation type; got %r' % (token,))
        command = token[1]
        rule = self.command_dispatch[command]
        token = next_val_fn()
        if token[0] != 'coordstart':
            raise SyntaxError("expecting '('; got %r" % (token,))
        numbers, token = rule(next_val_fn, token)
        if token[0] != 'coordend':
            raise SyntaxError("expecting ')'; got %r" % (token,))
        token = next_val_fn()
        return ((command, numbers), token)

    def rule_1or2numbers(self, next_val_fn, token):
        numbers = []
        token = next_val_fn()
        number, token = self.rule_number(next_val_fn, token)
        numbers.append(number)
        number, token = self.rule_optional_number(next_val_fn, token)
        if number is not None:
            numbers.append(number)
        return (numbers, token)

    def rule_1number(self, next_val_fn, token):
        token = next_val_fn()
        number, token = self.rule_number(next_val_fn, token)
        numbers = [number]
        return (numbers, token)

    def rule_1or3numbers(self, next_val_fn, token):
        numbers = []
        token = next_val_fn()
        number, token = self.rule_number(next_val_fn, token)
        numbers.append(number)
        number, token = self.rule_optional_number(next_val_fn, token)
        if number is not None:
            numbers.append(number)
            number, token = self.rule_number(next_val_fn, token)
            numbers.append(number)
        return (numbers, token)

    def rule_6numbers(self, next_val_fn, token):
        numbers = []
        token = next_val_fn()
        for i in range(6):
            number, token = self.rule_number(next_val_fn, token)
            numbers.append(number)
        return (numbers, token)

    def rule_number(self, next_val_fn, token):
        if token[0] not in self.number_tokens:
            raise SyntaxError('expecting a number; got %r' % (token,))
        x = Decimal(token[1]) * 1
        token = next_val_fn()
        return (x, token)

    def rule_optional_number(self, next_val_fn, token):
        if token[0] not in self.number_tokens:
            return (None, token)
        else:
            x = Decimal(token[1]) * 1
            token = next_val_fn()
            return (x, token)