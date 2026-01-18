from sympy.strategies.tools import subs, typed
from sympy.strategies.rl import rm_id
from sympy.core.basic import Basic
from sympy.core.singleton import S
def test_subs_empty():
    assert subs({})(Basic(S(1), S(2))) == Basic(S(1), S(2))