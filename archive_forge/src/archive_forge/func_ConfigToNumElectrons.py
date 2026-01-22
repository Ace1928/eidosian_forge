import os
import re
from rdkit import RDConfig
def ConfigToNumElectrons(config, ignoreFullD=0, ignoreFullF=0):
    """ counts the number of electrons appearing in a configuration string

      **Arguments**

        - config: the configuration string (e.g. '2s^2 2p^4')

        - ignoreFullD: toggles not counting full d shells

        - ignoreFullF: toggles not counting full f shells

      **Returns**

        the number of valence electrons

    """
    arr = config.split(' ')
    nEl = 0
    for i in range(1, len(arr)):
        l = arr[i].split('^')
        incr = int(l[1])
        if ignoreFullF and incr == 14 and (l[0].find('f') != -1) and (len(arr) > 2):
            incr = 0
        if ignoreFullD and incr == 10 and (l[0].find('d') != -1) and (len(arr) > 2):
            incr = 0
        nEl = nEl + incr
    return nEl