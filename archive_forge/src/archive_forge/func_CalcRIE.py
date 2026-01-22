import math
from collections import namedtuple
def CalcRIE(scores, col, alpha):
    """ RIE original definded here:
    Sheridan, R.P., Singh, S.B., Fluder, E.M. & Kearsley, S.K.
    Protocols for Bridging the Peptide to Nonpeptide Gap in Topological Similarity Searches.
    J. Chem. Inf. Comp. Sci. 41, 1395-1406 (2001).
    """
    RIE, _ = _RIEHelper(scores, col, alpha)
    return RIE