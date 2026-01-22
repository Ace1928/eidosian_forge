import numpy as np

    These results are from Bruce Hansen's Matlab package.
    To replicate the results, the exogenous variables were scaled
    down by 10**-2 and the results were then rescaled.

    These tests must also test likelihood functions because the
    llr when conducting hypothesis tests is the MLE while
    restricting the intercept to 0. Matlab's generic package always
    uses the unrestricted MLE.
    