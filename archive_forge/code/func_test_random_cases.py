import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_random_cases(self):
    self.optimize_compare('aab,fa,df,ecc->bde')
    self.optimize_compare('ecb,fef,bad,ed->ac')
    self.optimize_compare('bcf,bbb,fbf,fc->')
    self.optimize_compare('bb,ff,be->e')
    self.optimize_compare('bcb,bb,fc,fff->')
    self.optimize_compare('fbb,dfd,fc,fc->')
    self.optimize_compare('afd,ba,cc,dc->bf')
    self.optimize_compare('adb,bc,fa,cfc->d')
    self.optimize_compare('bbd,bda,fc,db->acf')
    self.optimize_compare('dba,ead,cad->bce')
    self.optimize_compare('aef,fbc,dca->bde')