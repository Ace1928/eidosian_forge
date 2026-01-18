from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
def testrepr(self):
    items = sorted(self.__dict__.items())
    print('. . . . . . . . .')
    for combo in items:
        print('  %s: %s' % combo)
    print('. . . . . . . . .')