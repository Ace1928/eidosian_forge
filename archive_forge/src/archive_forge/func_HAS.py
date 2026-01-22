from math import *
from rdkit import RDConfig
def HAS(strArg, composList, atomDict):
    """ *Calculator Method*

    does a string search

    **Arguments**

      - strArg: the arguments in string form

      - composList: the composition vector

      - atomDict: the atomic dictionary

    **Returns**

      1 or 0

  """
    splitArgs = strArg.split(',')
    if len(splitArgs) > 1:
        for atom, _ in composList:
            tStr = splitArgs[0].replace('DEADBEEF', atom)
            where = eval(tStr)
            what = eval(splitArgs[1])
            if what in where:
                return 1
        return 0
    else:
        return -666