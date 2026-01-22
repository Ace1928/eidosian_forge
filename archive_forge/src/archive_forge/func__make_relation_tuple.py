import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
@staticmethod
def _make_relation_tuple(position, values, num_entities):
    if len(values) == 1:
        return []
    else:
        sublist_size = len(values) // num_entities
        sublist_start = position // sublist_size
        sublist_position = int(position % sublist_size)
        sublist = values[sublist_start * sublist_size:(sublist_start + 1) * sublist_size]
        return [MaceCommand._make_model_var(sublist_start)] + MaceCommand._make_relation_tuple(sublist_position, sublist, num_entities)