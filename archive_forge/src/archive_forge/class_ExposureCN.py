import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
class ExposureCN(AbstractPropertyMap):
    """Residue exposure as number of CA atoms around its CA atom."""

    def __init__(self, model, radius=12.0, offset=0):
        """Initialize class.

        A residue's exposure is defined as the number of CA atoms around
        that residue's CA atom. A dictionary is returned that uses a L{Residue}
        object as key, and the residue exposure as corresponding value.

        :param model: the model that contains the residues
        :type model: L{Model}

        :param radius: radius of the sphere (centred at the CA atom)
        :type radius: float

        :param offset: number of flanking residues that are ignored in
                       the calculation of the number of neighbors
        :type offset: int

        """
        assert offset >= 0
        ppb = CaPPBuilder()
        ppl = ppb.build_peptides(model)
        fs_map = {}
        fs_list = []
        fs_keys = []
        for pp1 in ppl:
            for i in range(len(pp1)):
                fs = 0
                r1 = pp1[i]
                if not is_aa(r1) or not r1.has_id('CA'):
                    continue
                ca1 = r1['CA']
                for pp2 in ppl:
                    for j in range(len(pp2)):
                        if pp1 is pp2 and abs(i - j) <= offset:
                            continue
                        r2 = pp2[j]
                        if not is_aa(r2) or not r2.has_id('CA'):
                            continue
                        ca2 = r2['CA']
                        d = ca2 - ca1
                        if d < radius:
                            fs += 1
                res_id = r1.get_id()
                chain_id = r1.get_parent().get_id()
                fs_map[chain_id, res_id] = fs
                fs_list.append((r1, fs))
                fs_keys.append((chain_id, res_id))
                r1.xtra['EXP_CN'] = fs
        AbstractPropertyMap.__init__(self, fs_map, fs_keys, fs_list)