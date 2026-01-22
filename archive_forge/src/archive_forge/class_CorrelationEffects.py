import numpy as np
class CorrelationEffects(RegressionEffects):
    """
    Marginal correlation effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.

    Notes
    -----
    This class implements the marginal correlation approach to
    constructing test statistics for a knockoff analysis, as
    described under (1) in section 2.2 of the Barber and Candes
    paper.
    """

    def stats(self, parent):
        s1 = np.dot(parent.exog1.T, parent.endog)
        s2 = np.dot(parent.exog2.T, parent.endog)
        return np.abs(s1) - np.abs(s2)