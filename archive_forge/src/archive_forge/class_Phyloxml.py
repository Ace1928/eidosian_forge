import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Phyloxml(PhyloElement):
    """Root node of the PhyloXML document.

    Contains an arbitrary number of Phylogeny elements, possibly followed by
    elements from other namespaces.

    :Parameters:
        attributes : dict
            (XML namespace definitions)
        phylogenies : list
            The phylogenetic trees
        other : list
            Arbitrary non-phyloXML elements, if any

    """

    def __init__(self, attributes, phylogenies=None, other=None):
        """Initialize parameters for PhyloXML object."""
        self.attributes = {'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance', 'xmlns': 'http://www.phyloxml.org', 'xsi:schemaLocation': 'http://www.phyloxml.org http://www.phyloxml.org/1.10/phyloxml.xsd'}
        if attributes:
            self.attributes.update(attributes)
        self.phylogenies = phylogenies or []
        self.other = other or []

    def __getitem__(self, index):
        """Get a phylogeny by index or name."""
        if isinstance(index, (int, slice)):
            return self.phylogenies[index]
        if not isinstance(index, str):
            raise KeyError(f"can't use {type(index)} as an index")
        for tree in self.phylogenies:
            if tree.name == index:
                return tree
        else:
            raise KeyError(f'no phylogeny found with name {index!r}')

    def __iter__(self):
        """Iterate through the phylogenetic trees in this object."""
        return iter(self.phylogenies)

    def __len__(self):
        """Return the number of phylogenetic trees in this object."""
        return len(self.phylogenies)

    def __str__(self):
        """Return name of phylogenies in the object."""
        return '%s([%s])' % (self.__class__.__name__, ',\n'.join(map(str, self.phylogenies)))