import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis
class _AbstractHSExposure(AbstractPropertyMap):
    """Abstract class to calculate Half-Sphere Exposure (HSE).

    The HSE can be calculated based on the CA-CB vector, or the pseudo CB-CA
    vector based on three consecutive CA atoms. This is done by two separate
    subclasses.
    """

    def __init__(self, model, radius, offset, hse_up_key, hse_down_key, angle_key=None):
        """Initialize class.

        :param model: model
        :type model: L{Model}

        :param radius: HSE radius
        :type radius: float

        :param offset: number of flanking residues that are ignored in the
                       calculation of the number of neighbors
        :type offset: int

        :param hse_up_key: key used to store HSEup in the entity.xtra attribute
        :type hse_up_key: string

        :param hse_down_key: key used to store HSEdown in the entity.xtra attribute
        :type hse_down_key: string

        :param angle_key: key used to store the angle between CA-CB and CA-pCB in
                          the entity.xtra attribute
        :type angle_key: string
        """
        assert offset >= 0
        self.ca_cb_list = []
        ppb = CaPPBuilder()
        ppl = ppb.build_peptides(model)
        hse_map = {}
        hse_list = []
        hse_keys = []
        for pp1 in ppl:
            for i in range(len(pp1)):
                if i == 0:
                    r1 = None
                else:
                    r1 = pp1[i - 1]
                r2 = pp1[i]
                if i == len(pp1) - 1:
                    r3 = None
                else:
                    r3 = pp1[i + 1]
                result = self._get_cb(r1, r2, r3)
                if result is None:
                    continue
                pcb, angle = result
                hse_u = 0
                hse_d = 0
                ca2 = r2['CA'].get_vector()
                for pp2 in ppl:
                    for j in range(len(pp2)):
                        if pp1 is pp2 and abs(i - j) <= offset:
                            continue
                        ro = pp2[j]
                        if not is_aa(ro) or not ro.has_id('CA'):
                            continue
                        cao = ro['CA'].get_vector()
                        d = cao - ca2
                        if d.norm() < radius:
                            if d.angle(pcb) < pi / 2:
                                hse_u += 1
                            else:
                                hse_d += 1
                res_id = r2.get_id()
                chain_id = r2.get_parent().get_id()
                hse_map[chain_id, res_id] = (hse_u, hse_d, angle)
                hse_list.append((r2, (hse_u, hse_d, angle)))
                hse_keys.append((chain_id, res_id))
                r2.xtra[hse_up_key] = hse_u
                r2.xtra[hse_down_key] = hse_d
                if angle_key:
                    r2.xtra[angle_key] = angle
        AbstractPropertyMap.__init__(self, hse_map, hse_keys, hse_list)

    def _get_cb(self, r1, r2, r3):
        return NotImplemented

    def _get_gly_cb_vector(self, residue):
        """Return a pseudo CB vector for a Gly residue (PRIVATE).

        The pseudoCB vector is centered at the origin.

        CB coord=N coord rotated over -120 degrees
        along the CA-C axis.
        """
        try:
            n_v = residue['N'].get_vector()
            c_v = residue['C'].get_vector()
            ca_v = residue['CA'].get_vector()
        except Exception:
            return None
        n_v = n_v - ca_v
        c_v = c_v - ca_v
        rot = rotaxis(-pi * 120.0 / 180.0, c_v)
        cb_at_origin_v = n_v.left_multiply(rot)
        cb_v = cb_at_origin_v + ca_v
        self.ca_cb_list.append((ca_v, cb_v))
        return cb_at_origin_v