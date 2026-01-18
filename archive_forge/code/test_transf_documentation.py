import warnings # for silencing, see above...
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats, special
from statsmodels.sandbox.distributions.extras import (



Created on Sun May 09 22:35:21 2010
Author: josef-pktd
License: BSD

todo:
change moment calculation, (currently uses default _ppf method - I think)
# >>> lognormalg.moment(4)
Warning: The algorithm does not converge.  Roundoff error is detected
  in the extrapolation table.  It is assumed that the requested tolerance
  cannot be achieved, and that the returned result (if full_output = 1) is
  the best which can be obtained.
array(2981.0032380193438)
