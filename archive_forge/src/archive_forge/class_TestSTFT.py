import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
class TestSTFT:

    def test_input_validation(self):

        def chk_VE(match):
            """Assert for a ValueError matching regexp `match`.

            This little wrapper allows a more concise code layout.
            """
            return pytest.raises(ValueError, match=match)
        with chk_VE('nperseg must be a positive integer'):
            check_COLA('hann', -10, 0)
        with chk_VE('noverlap must be less than nperseg.'):
            check_COLA('hann', 10, 20)
        with chk_VE('window must be 1-D'):
            check_COLA(np.ones((2, 2)), 10, 0)
        with chk_VE('window must have length of nperseg'):
            check_COLA(np.ones(20), 10, 0)
        with chk_VE('nperseg must be a positive integer'):
            check_NOLA('hann', -10, 0)
        with chk_VE('noverlap must be less than nperseg'):
            check_NOLA('hann', 10, 20)
        with chk_VE('window must be 1-D'):
            check_NOLA(np.ones((2, 2)), 10, 0)
        with chk_VE('window must have length of nperseg'):
            check_NOLA(np.ones(20), 10, 0)
        with chk_VE('noverlap must be a nonnegative integer'):
            check_NOLA('hann', 64, -32)
        x = np.zeros(1024)
        z = stft(x)[2]
        with chk_VE('window must be 1-D'):
            stft(x, window=np.ones((2, 2)))
        with chk_VE('value specified for nperseg is different ' + 'from length of window'):
            stft(x, window=np.ones(10), nperseg=256)
        with chk_VE('nperseg must be a positive integer'):
            stft(x, nperseg=-256)
        with chk_VE('noverlap must be less than nperseg.'):
            stft(x, nperseg=256, noverlap=1024)
        with chk_VE('nfft must be greater than or equal to nperseg.'):
            stft(x, nperseg=256, nfft=8)
        with chk_VE('Input stft must be at least 2d!'):
            istft(x)
        with chk_VE('window must be 1-D'):
            istft(z, window=np.ones((2, 2)))
        with chk_VE('window must have length of 256'):
            istft(z, window=np.ones(10), nperseg=256)
        with chk_VE('nperseg must be a positive integer'):
            istft(z, nperseg=-256)
        with chk_VE('noverlap must be less than nperseg.'):
            istft(z, nperseg=256, noverlap=1024)
        with chk_VE('nfft must be greater than or equal to nperseg.'):
            istft(z, nperseg=256, nfft=8)
        with pytest.warns(UserWarning, match='NOLA condition failed, ' + 'STFT may not be invertible'):
            istft(z, nperseg=256, noverlap=0, window='hann')
        with chk_VE('Must specify differing time and frequency axes!'):
            istft(z, time_axis=0, freq_axis=0)
        with chk_VE('Unknown value for mode foo, must be one of: ' + "\\{'psd', 'stft'\\}"):
            _spectral_helper(x, x, mode='foo')
        with chk_VE("x and y must be equal if mode is 'stft'"):
            _spectral_helper(x[:512], x[512:], mode='stft')
        with chk_VE("Unknown boundary option 'foo', must be one of: " + "\\['even', 'odd', 'constant', 'zeros', None\\]"):
            _spectral_helper(x, x, boundary='foo')
        scaling = 'not_valid'
        with chk_VE(f"Parameter scaling={scaling!r} not in \\['spectrum', 'psd'\\]!"):
            stft(x, scaling=scaling)
        with chk_VE(f"Parameter scaling={scaling!r} not in \\['spectrum', 'psd'\\]!"):
            istft(z, scaling=scaling)

    def test_check_COLA(self):
        settings = [('boxcar', 10, 0), ('boxcar', 10, 9), ('bartlett', 51, 26), ('hann', 256, 128), ('hann', 256, 192), ('blackman', 300, 200), (('tukey', 0.5), 256, 64), ('hann', 256, 255)]
        for setting in settings:
            msg = '{}, {}, {}'.format(*setting)
            assert_equal(True, check_COLA(*setting), err_msg=msg)

    def test_check_NOLA(self):
        settings_pass = [('boxcar', 10, 0), ('boxcar', 10, 9), ('boxcar', 10, 7), ('bartlett', 51, 26), ('bartlett', 51, 10), ('hann', 256, 128), ('hann', 256, 192), ('hann', 256, 37), ('blackman', 300, 200), ('blackman', 300, 123), (('tukey', 0.5), 256, 64), (('tukey', 0.5), 256, 38), ('hann', 256, 255), ('hann', 256, 39)]
        for setting in settings_pass:
            msg = '{}, {}, {}'.format(*setting)
            assert_equal(True, check_NOLA(*setting), err_msg=msg)
        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings_fail = [(w_fail, len(w_fail), len(w_fail) // 2), ('hann', 64, 0)]
        for setting in settings_fail:
            msg = '{}, {}, {}'.format(*setting)
            assert_equal(False, check_NOLA(*setting), err_msg=msg)

    def test_average_all_segments(self):
        np.random.seed(1234)
        x = np.random.randn(1024)
        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8
        f, _, Z = stft(x, fs, window, nperseg, noverlap, padded=False, return_onesided=False, boundary=None)
        fw, Pw = welch(x, fs, window, nperseg, noverlap, return_onesided=False, scaling='spectrum', detrend=False)
        assert_allclose(f, fw)
        assert_allclose(np.mean(np.abs(Z) ** 2, axis=-1), Pw)

    def test_permute_axes(self):
        np.random.seed(1234)
        x = np.random.randn(1024)
        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8
        f1, t1, Z1 = stft(x, fs, window, nperseg, noverlap)
        f2, t2, Z2 = stft(x.reshape((-1, 1, 1)), fs, window, nperseg, noverlap, axis=0)
        t3, x1 = istft(Z1, fs, window, nperseg, noverlap)
        t4, x2 = istft(Z2.T, fs, window, nperseg, noverlap, time_axis=0, freq_axis=-1)
        assert_allclose(f1, f2)
        assert_allclose(t1, t2)
        assert_allclose(t3, t4)
        assert_allclose(Z1, Z2[:, 0, 0, :])
        assert_allclose(x1, x2[:, 0, 0])

    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    def test_roundtrip_real(self, scaling):
        np.random.seed(1234)
        settings = [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9), ('bartlett', 101, 51, 26), ('hann', 1024, 256, 128), (('tukey', 0.5), 1152, 256, 64), ('hann', 1024, 256, 255)]
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False, scaling=scaling)
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, scaling=scaling)
            msg = f'{window}, {noverlap}'
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)

    def test_roundtrip_not_nola(self):
        np.random.seed(1234)
        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings = [(w_fail, 256, len(w_fail), len(w_fail) // 2), ('hann', 256, 64, 0)]
        for window, N, nperseg, noverlap in settings:
            msg = f'{window}, {N}, {nperseg}, {noverlap}'
            assert not check_NOLA(window, nperseg, noverlap), msg
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary='zeros')
            with pytest.warns(UserWarning, match='NOLA'):
                tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, boundary=True)
            assert np.allclose(t, tr[:len(t)]), msg
            assert not np.allclose(x, xr[:len(x)]), msg

    def test_roundtrip_nola_not_cola(self):
        np.random.seed(1234)
        settings = [('boxcar', 100, 10, 3), ('bartlett', 101, 51, 37), ('hann', 1024, 256, 127), (('tukey', 0.5), 1152, 256, 14), ('hann', 1024, 256, 5)]
        for window, N, nperseg, noverlap in settings:
            msg = f'{window}, {nperseg}, {noverlap}'
            assert check_NOLA(window, nperseg, noverlap), msg
            assert not check_COLA(window, nperseg, noverlap), msg
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary='zeros')
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, boundary=True)
            msg = f'{window}, {noverlap}'
            assert_allclose(t, tr[:len(t)], err_msg=msg)
            assert_allclose(x, xr[:len(x)], err_msg=msg)

    def test_roundtrip_float32(self):
        np.random.seed(1234)
        settings = [('hann', 1024, 256, 128)]
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            x = x.astype(np.float32)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False)
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window)
            msg = f'{window}, {noverlap}'
            assert_allclose(t, t, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg, rtol=0.0001, atol=1e-05)
            assert_(x.dtype == xr.dtype)

    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    def test_roundtrip_complex(self, scaling):
        np.random.seed(1234)
        settings = [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9), ('bartlett', 101, 51, 26), ('hann', 1024, 256, 128), (('tukey', 0.5), 1152, 256, 64), ('hann', 1024, 256, 255)]
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size) + 10j * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False, return_onesided=False, scaling=scaling)
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, input_onesided=False, scaling=scaling)
            msg = f'{window}, {nperseg}, {noverlap}'
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Input data is complex, switching to return_onesided=False')
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=False, return_onesided=True, scaling=scaling)
        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, input_onesided=False, scaling=scaling)
        msg = f'{window}, {nperseg}, {noverlap}'
        assert_allclose(t, tr, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg)

    def test_roundtrip_boundary_extension(self):
        np.random.seed(1234)
        settings = [('boxcar', 100, 10, 0), ('boxcar', 100, 10, 9)]
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary=None)
            _, xr = istft(zz, noverlap=noverlap, window=window, boundary=False)
            for boundary in ['even', 'odd', 'constant', 'zeros']:
                _, _, zz_ext = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary=boundary)
                _, xr_ext = istft(zz_ext, noverlap=noverlap, window=window, boundary=True)
                msg = f'{window}, {noverlap}, {boundary}'
                assert_allclose(x, xr, err_msg=msg)
                assert_allclose(x, xr_ext, err_msg=msg)

    def test_roundtrip_padded_signal(self):
        np.random.seed(1234)
        settings = [('boxcar', 101, 10, 0), ('hann', 1000, 256, 128)]
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True)
            tr, xr = istft(zz, noverlap=noverlap, window=window)
            msg = f'{window}, {noverlap}'
            assert_allclose(t, tr[:t.size], err_msg=msg)
            assert_allclose(x, xr[:x.size], err_msg=msg)

    def test_roundtrip_padded_FFT(self):
        np.random.seed(1234)
        settings = [('hann', 1024, 256, 128, 512), ('hann', 1024, 256, 128, 501), ('boxcar', 100, 10, 0, 33), (('tukey', 0.5), 1152, 256, 64, 1024)]
        for window, N, nperseg, noverlap, nfft in settings:
            t = np.arange(N)
            x = 10 * np.random.randn(t.size)
            xc = x * np.exp(1j * np.pi / 4)
            _, _, z = stft(x, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, detrend=None, padded=True)
            _, _, zc = stft(xc, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, detrend=None, padded=True, return_onesided=False)
            tr, xr = istft(z, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window)
            tr, xcr = istft(zc, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, input_onesided=False)
            msg = f'{window}, {noverlap}'
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)
            assert_allclose(xc, xcr, err_msg=msg)

    def test_axis_rolling(self):
        np.random.seed(1234)
        x_flat = np.random.randn(1024)
        _, _, z_flat = stft(x_flat)
        for a in range(3):
            newshape = [1] * 3
            newshape[a] = -1
            x = x_flat.reshape(newshape)
            _, _, z_plus = stft(x, axis=a)
            _, _, z_minus = stft(x, axis=a - x.ndim)
            assert_equal(z_flat, z_plus.squeeze(), err_msg=a)
            assert_equal(z_flat, z_minus.squeeze(), err_msg=a - x.ndim)
        _, x_transpose_m = istft(z_flat.T, time_axis=-2, freq_axis=-1)
        _, x_transpose_p = istft(z_flat.T, time_axis=0, freq_axis=1)
        assert_allclose(x_flat, x_transpose_m, err_msg='istft transpose minus')
        assert_allclose(x_flat, x_transpose_p, err_msg='istft transpose plus')

    def test_roundtrip_scaling(self):
        """Verify behavior of scaling parameter. """
        X = np.zeros(513, dtype=complex)
        X[256] = 1024
        x = np.fft.irfft(X)
        power_x = sum(x ** 2) / len(x)
        Zs = stft(x, boundary='even', scaling='spectrum')[2]
        x1 = istft(Zs, boundary=True, scaling='spectrum')[1]
        assert_allclose(x1, x)
        assert_allclose(abs(Zs[63, :-1]), 0.5)
        assert_allclose(abs(Zs[64, :-1]), 1)
        assert_allclose(abs(Zs[65, :-1]), 0.5)
        Zs[63:66, :-1] = 0
        assert_allclose(Zs[:, :-1], 0, atol=np.finfo(Zs.dtype).resolution)
        Zp = stft(x, return_onesided=False, boundary='even', scaling='psd')[2]
        psd_Zp = np.sum(Zp.real ** 2 + Zp.imag ** 2, axis=0) / Zp.shape[0]
        assert_allclose(psd_Zp, power_x)
        x1 = istft(Zp, input_onesided=False, boundary=True, scaling='psd')[1]
        assert_allclose(x1, x)
        Zp0 = stft(x, return_onesided=True, boundary='even', scaling='psd')[2]
        Zp1 = np.conj(Zp0[-2:0:-1, :])
        assert_allclose(Zp[:129, :], Zp0)
        assert_allclose(Zp[129:, :], Zp1)
        s2 = np.sum(Zp0.real ** 2 + Zp0.imag ** 2, axis=0) + np.sum(Zp1.real ** 2 + Zp1.imag ** 2, axis=0)
        psd_Zp01 = s2 / (Zp0.shape[0] + Zp1.shape[0])
        assert_allclose(psd_Zp01, power_x)
        x1 = istft(Zp0, input_onesided=True, boundary=True, scaling='psd')[1]
        assert_allclose(x1, x)