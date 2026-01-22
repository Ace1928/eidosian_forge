import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
@deprecated('Use set_label() instead')
def _set_node(self, value):
    """Outdated method to set the node value; use the set_label() method instead."""