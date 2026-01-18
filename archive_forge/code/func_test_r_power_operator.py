import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_r_power_operator():
    seq_R = robjects.r['seq']
    mySeq = seq_R(0, 10)
    mySeqPow = mySeq.ro ** 2
    for i, li in enumerate(mySeq):
        assert li ** 2 == mySeqPow[i]