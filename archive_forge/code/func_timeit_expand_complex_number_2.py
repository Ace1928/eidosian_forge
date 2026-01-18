from sympy.core import symbols, I
def timeit_expand_complex_number_2():
    ((2 + 3 * I / 4) ** 1000).expand(complex=True)