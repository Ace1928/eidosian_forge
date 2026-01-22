from lib2to3 import fixer_base
from lib2to3.pytree import Node, Leaf
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Comma
Fixer for 'g.throw(E(V).with_traceback(T))' -> 'g.throw(E, V, T)'