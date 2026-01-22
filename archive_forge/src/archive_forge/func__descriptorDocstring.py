import bisect
import numpy
from rdkit.Chem.EState.EState import EStateIndices as EStateIndices_
from rdkit.Chem.MolSurf import _LabuteHelper as VSAContribs_
def _descriptorDocstring(name, nbin, bins):
    """ Create a docstring for the descriptor name """
    if nbin == 0:
        interval = '-inf < x <  {0:.2f}'.format(bins[nbin])
    elif nbin < len(bins):
        interval = ' {0:.2f} <= x <  {1:.2f}'.format(bins[nbin - 1], bins[nbin])
    else:
        interval = ' {0:.2f} <= x < inf'.format(bins[nbin - 1])
    return '{0} Descriptor {1} ({2})'.format(name, nbin + 1, interval)