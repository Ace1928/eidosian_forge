from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """
    ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """Initialize from stringPict and binding power."""
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning('\n                The unicode argument to prettyForm is deprecated. Only the s\n                argument (the first positional argument) should be passed.\n                ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
        self._unicode = unicode or s

    @property
    def unicode(self):
        sympy_deprecation_warning('\n            The prettyForm.unicode attribute is deprecated. Use the\n            prettyForm.s attribute instead.\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-pretty-printing-functions')
        return self._unicode

    def __add__(self, *others):
        """Make a pretty addition.
        Addition of negative numbers is simplified.
        """
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(*stringPict.next(*result), binding=prettyForm.ADD)

    def __truediv__(self, den, slashed=False):
        """Make a pretty division; stacked or slashed.
        """
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())
        if num.binding == prettyForm.NEG:
            num = num.right(' ')[0]
        return prettyForm(*stringPict.stack(num, stringPict.LINE, den), binding=prettyForm.DIV)

    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        quantity = {'degree': '°'}
        if len(others) == 0:
            return self
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            result.append(arg)
        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and (result[i + 1] == xsym('*')):
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        return prettyForm(*stringPict.next(*result), binding=bin)

    def __repr__(self):
        return 'prettyForm(%r,%d,%d)' % ('\n'.join(self.picture), self.baseline, self.binding)

    def __pow__(self, b):
        """Make a pretty power.
        """
        a = self
        use_inline_func_form = False
        if b.binding == prettyForm.POW:
            b = stringPict(*b.parens())
        if a.binding > prettyForm.FUNC:
            a = stringPict(*a.parens())
        elif a.binding == prettyForm.FUNC:
            if b.height() > 1:
                a = stringPict(*a.parens())
            else:
                use_inline_func_form = True
        if use_inline_func_form:
            b.baseline = a.prettyFunc.baseline + b.height()
            func = stringPict(*a.prettyFunc.right(b))
            return prettyForm(*func.right(a.prettyArgs))
        else:
            top = stringPict(*b.left(' ' * a.width()))
            bot = stringPict(*a.right(' ' * b.width()))
        return prettyForm(*bot.above(top), binding=prettyForm.POW)
    simpleFunctions = ['sin', 'cos', 'tan']

    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        if function in prettyForm.simpleFunctions:
            assert len(args) == 1, 'Simple function %s must have 1 argument' % function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                return prettyForm(*arg.left(function + ' '), binding=prettyForm.FUNC)
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(*argumentList.left(function), binding=prettyForm.ATOM)