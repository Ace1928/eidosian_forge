import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
@property
def _special_function_classes(self):
    from sympy.functions.special.tensor_functions import KroneckerDelta
    from sympy.functions.special.gamma_functions import gamma, lowergamma
    from sympy.functions.special.zeta_functions import lerchphi
    from sympy.functions.special.beta_functions import beta
    from sympy.functions.special.delta_functions import DiracDelta
    from sympy.functions.special.error_functions import Chi
    return {KroneckerDelta: [greek_unicode['delta'], 'delta'], gamma: [greek_unicode['Gamma'], 'Gamma'], lerchphi: [greek_unicode['Phi'], 'lerchphi'], lowergamma: [greek_unicode['gamma'], 'gamma'], beta: [greek_unicode['Beta'], 'B'], DiracDelta: [greek_unicode['delta'], 'delta'], Chi: ['Chi', 'Chi']}