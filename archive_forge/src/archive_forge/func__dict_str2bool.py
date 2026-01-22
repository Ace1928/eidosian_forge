from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _dict_str2bool(dct, keys):
    """Return a new dictionary where string values are replaced with booleans (PRIVATE)."""
    out = dct.copy()
    for key in keys:
        if key in out:
            out[key] = _str2bool(out[key])
    return out