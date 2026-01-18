from collections import defaultdict
from functools import reduce
from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
def unique_names_demo():
    lexpr = Expression.fromstring
    p1 = lexpr('man(Socrates)')
    p2 = lexpr('man(Bill)')
    c = lexpr('exists x.exists y.(x != y)')
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print('assumptions:')
    for a in unp.assumptions():
        print('   ', a)
    print('goal:', unp.goal())
    print(unp.prove())
    p1 = lexpr('all x.(walk(x) -> (x = Socrates))')
    p2 = lexpr('Bill = William')
    p3 = lexpr('Bill = Billy')
    c = lexpr('-walk(William)')
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print('assumptions:')
    for a in unp.assumptions():
        print('   ', a)
    print('goal:', unp.goal())
    print(unp.prove())