from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class SemitoneBandpassFilterbank(object):
    """
    Time domain semitone filterbank of elliptic filters as proposed in [1]_.

    Parameters
    ----------
    order : int, optional
        Order of elliptic filters.
    passband_ripple : float, optional
        Maximum ripple allowed below unity gain in the passband [dB].
    stopband_rejection : float, optional
        Minimum attenuation required in the stop band [dB].
    q_factor : int, optional
        Q-factor of the filters.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Reference frequency for the first bandpass filter [Hz].

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    Notes
    -----
    This is a time domain filterbank, thus it cannot be used as the other
    time-frequency filterbanks of this module. Instead of ``np.dot()`` use
    ``scipy.signal.filtfilt()`` to filter a signal.

    """

    def __init__(self, order=4, passband_ripple=1, stopband_rejection=50, q_factor=25, fmin=27.5, fmax=4200.0, fref=A4):
        from scipy.signal import ellip
        self.order = order
        self.passband_ripple = passband_ripple
        self.stopband_rejection = stopband_rejection
        self.q_factor = q_factor
        self.fref = fref
        self.center_frequencies = semitone_frequencies(fmin, fmax, fref=fref)
        self.band_sample_rates = np.ones_like(self.center_frequencies) * 4410
        self.band_sample_rates[self.center_frequencies > 2000] = 22050
        self.band_sample_rates[self.center_frequencies < 250] = 882
        self.filters = []
        for freq, sample_rate in zip(self.center_frequencies, self.band_sample_rates):
            freqs = [(freq - freq / q_factor / 2.0) * 2.0 / sample_rate, (freq + freq / q_factor / 2.0) * 2.0 / sample_rate]
            self.filters.append(ellip(order, passband_ripple, stopband_rejection, freqs, btype='bandpass'))

    @property
    def num_bands(self):
        """Number of bands."""
        return len(self.center_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        f = self.center_frequencies[0]
        return f - f / self.q_factor / 2.0

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        f = self.center_frequencies[-1]
        return f + f / self.q_factor / 2.0