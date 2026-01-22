from scipy.datasets._registry import registry
from scipy.datasets._fetchers import data_fetcher
from scipy.datasets._utils import _clear_cache
from scipy.datasets import ascent, face, electrocardiogram, download_all
from numpy.testing import assert_equal, assert_almost_equal
import os
import pytest
class TestDatasets:

    @pytest.fixture(scope='module', autouse=True)
    def test_download_all(self):
        download_all()
        yield

    def test_existence_all(self):
        assert len(os.listdir(data_dir)) >= len(registry)

    def test_ascent(self):
        assert_equal(ascent().shape, (512, 512))
        assert _has_hash(os.path.join(data_dir, 'ascent.dat'), registry['ascent.dat'])

    def test_face(self):
        assert_equal(face().shape, (768, 1024, 3))
        assert _has_hash(os.path.join(data_dir, 'face.dat'), registry['face.dat'])

    def test_electrocardiogram(self):
        ecg = electrocardiogram()
        assert_equal(ecg.dtype, float)
        assert_equal(ecg.shape, (108000,))
        assert_almost_equal(ecg.mean(), -0.16510875)
        assert_almost_equal(ecg.std(), 0.5992473991177294)
        assert _has_hash(os.path.join(data_dir, 'ecg.dat'), registry['ecg.dat'])