import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def to_glueformula_list(self, depgraph, node=None, counter=None, verbose=False):
    if node is None:
        top = depgraph.nodes[0]
        depList = list(chain.from_iterable(top['deps'].values()))
        root = depgraph.nodes[depList[0]]
        return self.to_glueformula_list(depgraph, root, Counter(), verbose)
    glueformulas = self.lookup(node, depgraph, counter)
    for dep_idx in chain.from_iterable(node['deps'].values()):
        dep = depgraph.nodes[dep_idx]
        glueformulas.extend(self.to_glueformula_list(depgraph, dep, counter, verbose))
    return glueformulas