import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
class StereoEnumerationOptions(object):
    """
          - tryEmbedding: if set the process attempts to generate a standard RDKit distance geometry
            conformation for the stereisomer. If this fails, we assume that the stereoisomer is
            non-physical and don't return it. NOTE that this is computationally expensive and is
            just a heuristic that could result in stereoisomers being lost.

          - onlyUnassigned: if set (the default), stereocenters which have specified stereochemistry
            will not be perturbed unless they are part of a relative stereo
            group.

          - maxIsomers: the maximum number of isomers to yield, if the
            number of possible isomers is greater than maxIsomers, a
            random subset will be yielded. If 0, all isomers are
            yielded. Since every additional stereo center doubles the
            number of results (and execution time) it's important to
            keep an eye on this.

          - onlyStereoGroups: Only find stereoisomers that differ at the
            StereoGroups associated with the molecule.
    """
    __slots__ = ('tryEmbedding', 'onlyUnassigned', 'onlyStereoGroups', 'maxIsomers', 'rand', 'unique')

    def __init__(self, tryEmbedding=False, onlyUnassigned=True, maxIsomers=1024, rand=None, unique=True, onlyStereoGroups=False):
        self.tryEmbedding = tryEmbedding
        self.onlyUnassigned = onlyUnassigned
        self.onlyStereoGroups = onlyStereoGroups
        self.maxIsomers = maxIsomers
        self.rand = rand
        self.unique = unique