from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
class TestGridDosData:
    """Test the grid DOS data container"""

    def test_init(self):
        with pytest.raises(ValueError):
            GridDOSData(np.linspace(0, 10, 11), np.zeros(10))
        with pytest.raises(ValueError):
            GridDOSData(np.linspace(0, 10, 11) ** 2, np.zeros(11))

    @pytest.fixture
    def dense_dos(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.sin(x / 10)
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2s', 'day': 'Tue'})

    @pytest.fixture
    def denser_dos(self):
        x = np.linspace(0.0, 10.0, 21)
        y = np.sin(x / 10)
        return GridDOSData(x, y)

    @pytest.fixture
    def another_dense_dos(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.sin(x / 10) * 2
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2p', 'month': 'Feb'})

    def test_access(self, dense_dos):
        assert dense_dos.info == {'symbol': 'C', 'orbital': '2s', 'day': 'Tue'}
        assert len(dense_dos.get_energies()) == 11
        assert dense_dos.get_energies()[-2] == pytest.approx(9.0)
        assert dense_dos.get_weights()[-1] == pytest.approx(np.sin(1))

    def test_copy(self, dense_dos):
        copy_dos = dense_dos.copy()
        assert copy_dos.info == dense_dos.info
        dense_dos.info['symbol'] = 'X'
        assert dense_dos.info['symbol'] == 'X'
        assert copy_dos.info['symbol'] == 'C'

    def test_addition(self, dense_dos, another_dense_dos):
        sum_dos = dense_dos + another_dense_dos
        assert np.allclose(sum_dos.get_energies(), dense_dos.get_energies())
        assert np.allclose(sum_dos.get_weights(), dense_dos.get_weights() * 3)
        assert sum_dos.info == {'symbol': 'C'}
        with pytest.raises(ValueError):
            dense_dos + GridDOSData(dense_dos.get_energies() + 1.0, dense_dos.get_weights())
        with pytest.raises(ValueError):
            dense_dos + GridDOSData(dense_dos.get_energies()[1:], dense_dos.get_weights()[1:])

    def test_check_spacing(self, dense_dos):
        """Check a warning is logged when width < 2 * grid spacing"""
        dense_dos._sample([1], width=2.1)
        with pytest.warns(UserWarning, match='The broadening width is small'):
            dense_dos._sample([1], width=1.9)

    def test_resampling_consistency(self, dense_dos, denser_dos):
        """Check that resampled spectra are independent of the original density

        Compare resampling of sample function on two different grids to the
        same new grid with broadening. We accept a 5% difference because the
        initial shape is slightly different; what we are checking for is a
        factor 2 difference from "double-counting" the extra data points.
        """
        sampling_params = dict(npts=500, xmin=0, xmax=10, width=4)
        from_dense = dense_dos.sample_grid(**sampling_params)
        from_denser = denser_dos.sample_grid(**sampling_params)
        assert np.allclose(from_dense.get_energies(), from_denser.get_energies())
        assert np.allclose(from_dense.get_weights(), from_denser.get_weights(), rtol=0.05, atol=0.01)
    linewidths = [1, 5, None]

    @pytest.mark.usefixtures('figure')
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot(self, dense_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}
        ax = figure.add_subplot(111)
        ax_out = dense_dos.plot(ax=ax, mplargs=mplargs, smearing='Gauss')
        assert ax_out == ax
        line_data = ax.lines[0].get_data()
        assert np.allclose(line_data[0], np.linspace(0.0, 10.0, 11))
        assert np.allclose(line_data[1], np.sin(np.linspace(0.0, 1.0, 11)))

    @pytest.mark.usefixtures('figure')
    def test_plot_broad_dos(self, dense_dos, figure):
        ax = figure.add_subplot(111)
        _ = dense_dos.plot(ax=ax, npts=10, xmin=0, xmax=9, width=4, smearing='Gauss')
        line_data = ax.lines[0].get_data()
        assert np.allclose(line_data[0], range(10))
        assert np.allclose(line_data[1], [0.14659725, 0.19285644, 0.24345501, 0.29505574, 0.34335948, 0.38356488, 0.41104823, 0.42216901, 0.41503382, 0.39000808])
    smearing_args = [(dict(npts=0, width=None), (0, None)), (dict(npts=10, width=None, default_width=5.0), (10, 5.0)), (dict(npts=0, width=0.5, default_npts=100), (100, 0.5)), (dict(npts=10, width=0.5), (10, 0.5))]

    @pytest.mark.parametrize('inputs, expected', smearing_args)
    def test_smearing_args_interpreter(self, inputs, expected):
        assert GridDOSData._interpret_smearing_args(**inputs) == expected