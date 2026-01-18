from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel

        method : LSDV, demeaned, MLE, GLS, BE, FE, optional
        model :
                between
                fixed
                random
                pooled
                [gmm]
        effects :
                oneway
                time
                twoway
        femethod : demeaned (only one implemented)
                   WLS
        remethod :
                swar -
                amemiya
                nerlove
                walhus


        Notes
        -----
        This is unfinished.  None of the method arguments work yet.
        Only oneway effects should work.
        