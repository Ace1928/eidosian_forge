import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
class TestFindPeaksCwt:

    def test_find_peaks_exact(self):
        """
        Generate a series of gaussians and attempt to find the peak locations.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0, min_length=None)
        np.testing.assert_array_equal(found_locs, act_locs, 'Found maximum locations did not equal those expected')

    def test_find_peaks_withnoise(self):
        """
        Verify that peak locations are (approximately) found
        for a series of gaussians with added noise.
        """
        sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
        num_points = 500
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas))
        noise_amp = 0.07
        np.random.seed(18181911)
        test_data += (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
        found_locs = find_peaks_cwt(test_data, widths, min_length=15, gap_thresh=1, min_snr=noise_amp / 5)
        np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' + 'of peaks found than expected')
        diffs = np.abs(found_locs - act_locs)
        max_diffs = np.array(sigmas) / 5
        np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' + 'by more than %s' % max_diffs)

    def test_find_peaks_nopeak(self):
        """
        Verify that no peak is found in
        data that's just noise.
        """
        noise_amp = 1.0
        num_points = 100
        np.random.seed(181819141)
        test_data = (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
        widths = np.arange(10, 50)
        found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
        np.testing.assert_equal(len(found_locs), 0)

    def test_find_peaks_with_non_default_wavelets(self):
        x = gaussian(200, 2)
        widths = np.array([1, 2, 3, 4])
        a = find_peaks_cwt(x, widths, wavelet=gaussian)
        np.testing.assert_equal(np.array([100]), a)

    def test_find_peaks_window_size(self):
        """
        Verify that window_size is passed correctly to private function and
        affects the result.
        """
        sigmas = [2.0, 2.0]
        num_points = 1000
        test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
        widths = np.arange(0.1, max(sigmas), 0.2)
        noise_amp = 0.05
        np.random.seed(18181911)
        test_data += (np.random.rand(num_points) - 0.5) * (2 * noise_amp)
        test_data[250:320] -= 1
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3, min_length=None, window_size=None)
        with pytest.raises(AssertionError):
            assert found_locs.size == act_locs.size
        found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=3, min_length=None, window_size=20)
        assert found_locs.size == act_locs.size

    def test_find_peaks_with_one_width(self):
        """
        Verify that the `width` argument
        in `find_peaks_cwt` can be a float
        """
        xs = np.arange(0, np.pi, 0.05)
        test_data = np.sin(xs)
        widths = 1
        found_locs = find_peaks_cwt(test_data, widths)
        np.testing.assert_equal(found_locs, 32)