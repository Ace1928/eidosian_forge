from copy import copy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SCOP.Residues import Residues
class SeqMapIndex(dict):
    """An RAF file index.

    The RAF file itself is about 50 MB. This index provides rapid, random
    access of RAF records without having to load the entire file into memory.

    The index key is a concatenation of the  PDB ID and chain ID. e.g
    "2drcA", ``"155c_"``. RAF uses an underscore to indicate blank
    chain IDs.
    """

    def __init__(self, filename):
        """Initialize the RAF file index.

        Arguments:
         - filename  -- The file to index

        """
        dict.__init__(self)
        self.filename = filename
        with open(self.filename) as f:
            position = 0
            while True:
                line = f.readline()
                if not line:
                    break
                key = line[0:5]
                if key is not None:
                    self[key] = position
                position = f.tell()

    def __getitem__(self, key):
        """Return an item from the indexed file."""
        position = dict.__getitem__(self, key)
        with open(self.filename) as f:
            f.seek(position)
            line = f.readline()
            record = SeqMap(line)
        return record

    def getSeqMap(self, residues):
        """Get the sequence map for a collection of residues.

        Arguments:
         - residues -- A Residues instance, or a string that can be
           converted into a Residues instance.

        """
        if isinstance(residues, str):
            residues = Residues(residues)
        pdbid = residues.pdbid
        frags = residues.fragments
        if not frags:
            frags = (('_', '', ''),)
        seqMap = None
        for frag in frags:
            chainid = frag[0]
            if chainid in ['', '-', ' ', '_']:
                chainid = '_'
            id = pdbid + chainid
            sm = self[id]
            start = 0
            end = len(sm.res)
            if frag[1]:
                start = int(sm.index(frag[1], chainid))
            if frag[2]:
                end = int(sm.index(frag[2], chainid)) + 1
            sm = sm[start:end]
            if seqMap is None:
                seqMap = sm
            else:
                seqMap += sm
        return seqMap