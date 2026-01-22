import collections
import copy
import itertools
import random
import re
import warnings
def as_phyloxml(self, **kwargs):
    """Convert this tree to a PhyloXML-compatible Phylogeny.

        This lets you use the additional annotation types PhyloXML defines, and
        save this information when you write this tree as 'phyloxml'.
        """
    from Bio.Phylo.PhyloXML import Phylogeny
    return Phylogeny.from_tree(self, **kwargs)