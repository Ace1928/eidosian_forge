import os
import numpy as np
class ARResultsMLE:
    """
    Results of fitting an AR(9) model to the sunspot data using exact MLE.

    Results were taken from gretl.
    """

    def __init__(self, constant=True):
        self.avobs = 300
        if constant:
            filename = os.path.join(cur_dir, 'ARMLEConstantPredict.csv')
            filename2 = os.path.join(cur_dir, 'results_ar_forecast_mle_dynamic.csv')
            predictresults = np.loadtxt(filename, delimiter=',')
            pv = predictresults[:, 1]
            dynamicpv = np.genfromtxt(filename2, delimiter=',', skip_header=1)
            self.FVMLEdefault = pv[:309]
            self.FVMLEstart9end308 = pv[9:309]
            self.FVMLEstart100end308 = pv[100:309]
            self.FVMLEstart0end200 = pv[:201]
            self.FVMLEstart200end334 = pv[200:]
            self.FVMLEstart308end334 = pv[308:]
            self.FVMLEstart9end309 = pv[9:310]
            self.FVMLEstart0end301 = pv[:302]
            self.FVMLEstart4end312 = pv[4:313]
            self.FVMLEstart2end7 = pv[2:8]
            self.fcdyn = dynamicpv[:, 0]
            self.fcdyn2 = dynamicpv[:, 1]
            self.fcdyn3 = dynamicpv[:, 2]
            self.fcdyn4 = dynamicpv[:, 3]
        else:
            pass