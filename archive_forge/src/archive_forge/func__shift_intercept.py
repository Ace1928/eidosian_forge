import numpy as np
def _shift_intercept(arr):
    """
    A convenience function to make the SAS covariance matrix
    compatible with statsmodels.rlm covariance
    """
    arr = np.asarray(arr)
    side = int(np.sqrt(len(arr)))
    return np.roll(np.roll(arr.reshape(side, side), -1, axis=1), -1, axis=0)