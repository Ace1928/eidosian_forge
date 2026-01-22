import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def _spddirect2(self, n):
    """this looks bad, maybe with an fftshift
        """
    hw = fft.fft(np.r_[self.ma[::-1], self.ma], n) / fft.fft(np.r_[self.ar[::-1], self.ar], n)
    return hw * hw.conj()