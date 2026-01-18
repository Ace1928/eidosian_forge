from numpy.testing import assert_equal
import numpy as np
def ttest_interaction(self):
    """ttests for no-interaction terms are zero
        """
    nia = self.n_interaction
    R_nointer = np.hstack((np.zeros((nia, self.nvars - nia)), np.eye(nia)))
    R_nointer_transf = self.transform.inv_dot_right(R_nointer)
    self.R_nointer_transf = R_nointer_transf
    t_res = self.resols.t_test(R_nointer_transf)
    return t_res