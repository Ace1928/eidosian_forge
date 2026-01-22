from jmespath.compat import with_str_method
@with_str_method
class ArityError(ParseError):

    def __init__(self, expected, actual, name):
        self.expected_arity = expected
        self.actual_arity = actual
        self.function_name = name
        self.expression = None

    def __str__(self):
        return 'Expected %s %s for function %s(), received %s' % (self.expected_arity, self._pluralize('argument', self.expected_arity), self.function_name, self.actual_arity)

    def _pluralize(self, word, count):
        if count == 1:
            return word
        else:
            return word + 's'