from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
class TestRawDosData:
    """Test the raw DOS data container"""

    @pytest.fixture
    def sparse_dos(self):
        return RawDOSData([1.2, 3.4, 5.0], [3.0, 2.1, 0.0], info={'symbol': 'H', 'number': '1', 'food': 'egg'})

    @pytest.fixture
    def another_sparse_dos(self):
        return RawDOSData([8.0, 2.0, 2.0, 5.0], [1.0, 1.0, 1.0, 1.0], info={'symbol': 'H', 'number': '2'})

    def test_init(self):
        with pytest.raises(ValueError):
            RawDOSData([1, 2, 3], [4, 5], info={'symbol': 'H'})

    def test_access(self, sparse_dos):
        assert sparse_dos.info == {'symbol': 'H', 'number': '1', 'food': 'egg'}
        assert np.allclose(sparse_dos.get_energies(), [1.2, 3.4, 5.0])
        assert np.allclose(sparse_dos.get_weights(), [3.0, 2.1, 0.0])

    def test_copy(self, sparse_dos):
        copy_dos = sparse_dos.copy()
        assert copy_dos.info == sparse_dos.info
        sparse_dos.info['symbol'] = 'X'
        assert sparse_dos.info['symbol'] == 'X'
        assert copy_dos.info['symbol'] == 'H'

    @pytest.mark.parametrize('other', [True, 1, 0.5, 'string'])
    def test_equality_wrongtype(self, sparse_dos, other):
        assert not sparse_dos._almost_equals(other)
    equality_data = [(((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), True), (((1.0, 3.0), (3.0, 4.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), False), (((1.0, 2.0), (3.0, 5.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), False), (((1.0, 3.0), (3.0, 5.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), False), (((1.0, 2.0), (3.0, 4.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'He'}), False), (((1.0, 3.0), (3.0, 4.0), {'symbol': 'H'}), ((1.0, 2.0), (3.0, 4.0), {'symbol': 'He'}), False)]

    @pytest.mark.parametrize('data_1, data_2, isequal', equality_data)
    def test_equality(self, data_1, data_2, isequal):
        assert RawDOSData(*data_1[:2], info=data_1[2])._almost_equals(RawDOSData(*data_2[:2], info=data_2[2])) == isequal

    def test_addition(self, sparse_dos, another_sparse_dos):
        summed_dos = sparse_dos + another_sparse_dos
        assert summed_dos.info == {'symbol': 'H'}
        assert np.allclose(summed_dos.get_energies(), [1.2, 3.4, 5.0, 8.0, 2.0, 2.0, 5.0])
        assert np.allclose(summed_dos.get_weights(), [3.0, 2.1, 0.0, 1.0, 1.0, 1.0, 1.0])
    sampling_data_args_results = [([[0.0], [1.0]], [[0.0], {'width': 1}], [1.0 / np.sqrt(2.0 * np.pi)]), ([[1.0], [2.0]], [[1.0], {'width': 0.5}], [2.0 / (np.sqrt(2.0 * np.pi) * 0.5)]), ([[1.0, 1.0], [2.0, 1.0]], [[1.0], {'width': 1}], [3.0 / np.sqrt(2.0 * np.pi)]), ([[0.0], [1.0]], [[np.sqrt(2 * np.log(2)) * 3], {'width': 3}], [0.5 / (np.sqrt(2 * np.pi) * 3)]), ([[1.2, 3.4, 5], [3.0, 2.1, 0.0]], [[1.0, 1.5, 2.0, 2.4], {'width': 2}], [0.79932418, 0.85848101, 0.88027184, 0.8695055])]

    @pytest.mark.parametrize('data, args, result', sampling_data_args_results)
    def test_sampling(self, data, args, result):
        dos = RawDOSData(data[0], data[1])
        weights = dos._sample(*args[:-1], **args[-1])
        assert np.allclose(weights, result)
        with pytest.raises(ValueError):
            dos._sample([1], smearing="Gauss's spherical cousin")

    def test_sampling_error(self, sparse_dos):
        with pytest.raises(ValueError):
            sparse_dos._sample([1, 2, 3], width=0.0)
        with pytest.raises(ValueError):
            sparse_dos._sample([1, 2, 3], width=-1)

    def test_sample_grid(self, sparse_dos):
        min_dos = sparse_dos.sample_grid(10, xmax=5, padding=3, width=0.1)
        assert min_dos.get_energies()[0] == pytest.approx(1.2 - 3 * 0.1)
        max_dos = sparse_dos.sample_grid(10, xmin=0, padding=2, width=0.2)
        assert max_dos.get_energies()[-1] == pytest.approx(5 + 2 * 0.2)
        default_dos = sparse_dos.sample_grid(10)
        assert np.allclose(default_dos.get_energies(), np.linspace(0.9, 5.3, 10))
        dos0 = sparse_dos._sample(np.linspace(0.9, 5.3, 10))
        assert np.allclose(default_dos.get_weights(), dos0)
    linewidths = [1, 5, None]

    @pytest.mark.usefixtures('figure')
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot(self, sparse_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}
        ax = figure.add_subplot(111)
        ax_out = sparse_dos.plot(npts=5, ax=ax, mplargs=mplargs, smearing='Gauss')
        assert ax_out == ax
        line_data = ax.lines[0].get_data()
        assert np.allclose(line_data[0], np.linspace(0.9, 5.3, 5))
        assert np.allclose(line_data[1], [0.132955452, 1.51568133e-13, 0.0930688167, 1.06097693e-13, 3.41173568e-78])

    @pytest.mark.usefixtures('figure')
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot_deltas(self, sparse_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}
        ax = figure.add_subplot(111)
        ax_out = sparse_dos.plot_deltas(ax=ax, mplargs=mplargs)
        assert ax_out == ax
        assert np.allclose(list(map(lambda x: x.vertices, ax.get_children()[0].get_paths())), [[[1.2, 0.0], [1.2, 3.0]], [[3.4, 0.0], [3.4, 2.1]], [[5.0, 0.0], [5.0, 0.0]]])