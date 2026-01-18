import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def test_transform_output(argument_pair):
    """
    Transform the model into various Mace4 ``interpformat`` formats.
    """
    g = Expression.fromstring(argument_pair[0])
    alist = [lp.parse(a) for a in argument_pair[1]]
    m = MaceCommand(g, assumptions=alist)
    m.build_model()
    for a in alist:
        print('   %s' % a)
    print(f'|- {g}: {m.build_model()}\n')
    for format in ['standard', 'portable', 'xml', 'cooked']:
        spacer()
        print("Using '%s' format" % format)
        spacer()
        print(m.model(format=format))