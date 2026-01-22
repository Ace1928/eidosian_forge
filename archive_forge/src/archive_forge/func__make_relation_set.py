import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
@staticmethod
def _make_relation_set(num_entities, values):
    """
        Convert a Mace4-style relation table into a dictionary.

        :param num_entities: the number of entities in the model; determines the row length in the table.
        :type num_entities: int
        :param values: a list of 1's and 0's that represent whether a relation holds in a Mace4 model.
        :type values: list of int
        """
    r = set()
    for position in [pos for pos, v in enumerate(values) if v == 1]:
        r.add(tuple(MaceCommand._make_relation_tuple(position, values, num_entities)))
    return r