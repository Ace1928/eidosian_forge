import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_r_mult_operator():
    seq_R = robjects.r['seq']
    mySeq = seq_R(0, 10)
    mySeqAdd = mySeq.ro * 2
    for i, li in enumerate(mySeq):
        assert li * 2 == mySeqAdd[i]