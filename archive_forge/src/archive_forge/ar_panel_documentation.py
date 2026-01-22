import numpy as np
from scipy import optimize
from statsmodels.regression.linear_model import OLS
Paneldata model with fixed effect (constants) and AR(1) errors

checking fast evaluation of groupar1filter
quickly written to try out grouparfilter without python loops

maybe the example has MA(1) not AR(1) errors, I'm not sure and changed this.

results look good, I'm also differencing the dummy variable (constants) ???
e.g. nobs = 35
true 0.6, 10, 20, 30   (alpha, mean_0, mean_1, mean_2)
estimate 0.369453125 [ 10.14646929  19.87135086  30.12706505]

Currently minimizes ssr but could switch to minimize llf, i.e. conditional MLE.
This should correspond to iterative FGLS, where data are AR(1) transformed
similar to GLSAR ?
Result statistic from GLS return by OLS on transformed data should be
asymptotically correct (check)

Could be extended to AR(p) errors, but then requires panel with larger T

