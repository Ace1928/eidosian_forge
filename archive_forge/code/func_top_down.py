from itertools import product
from sympy.strategies.util import basic_fns
from .core import chain, identity, do_one
def top_down(brule, fns=basic_fns):
    """ Apply a rule down a tree running it on the top nodes first """
    return chain(do_one(brule, identity), lambda expr: sall(top_down(brule, fns), fns)(expr))