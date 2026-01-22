import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
class TestGLCM:

    def setup_method(self):
        self.image = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 3, 3]], dtype=np.uint8)

    @run_in_parallel()
    def test_output_angles(self):
        result = graycomatrix(self.image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 4)
        assert result.shape == (4, 4, 1, 4)
        expected1 = np.array([[2, 2, 1, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
        expected2 = np.array([[1, 1, 3, 0], [0, 1, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 1], expected2)
        expected3 = np.array([[3, 0, 2, 0], [0, 2, 2, 0], [0, 0, 1, 2], [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 2], expected3)
        expected4 = np.array([[2, 0, 0, 0], [1, 1, 2, 0], [0, 0, 2, 1], [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 3], expected4)

    def test_output_symmetric_1(self):
        result = graycomatrix(self.image, [1], [np.pi / 2], 4, symmetric=True)
        assert result.shape == (4, 4, 1, 1)
        expected = np.array([[6, 0, 2, 0], [0, 4, 2, 0], [2, 2, 2, 2], [0, 0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_error_raise_float(self):
        for dtype in [float, np.double, np.float16, np.float32, np.float64]:
            with pytest.raises(ValueError):
                graycomatrix(self.image.astype(dtype), [1], [np.pi], 4)

    def test_error_raise_int_types(self):
        for dtype in [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]:
            with pytest.raises(ValueError):
                graycomatrix(self.image.astype(dtype), [1], [np.pi])

    def test_error_raise_negative(self):
        with pytest.raises(ValueError):
            graycomatrix(self.image.astype(np.int16) - 1, [1], [np.pi], 4)

    def test_error_raise_levels_smaller_max(self):
        with pytest.raises(ValueError):
            graycomatrix(self.image - 1, [1], [np.pi], 3)

    def test_image_data_types(self):
        for dtype in [np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64]:
            img = self.image.astype(dtype)
            result = graycomatrix(img, [1], [np.pi / 2], 4, symmetric=True)
            assert result.shape == (4, 4, 1, 1)
            expected = np.array([[6, 0, 2, 0], [0, 4, 2, 0], [2, 2, 2, 2], [0, 0, 2, 0]], dtype=np.uint32)
            np.testing.assert_array_equal(result[:, :, 0, 0], expected)
        return

    def test_output_distance(self):
        im = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 2], [3, 0, 0, 3]], dtype=np.uint8)
        result = graycomatrix(im, [3], [0], 4, symmetric=False)
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_output_combo(self):
        im = np.array([[0], [1], [2], [3]], dtype=np.uint8)
        result = graycomatrix(im, [1, 2], [0, np.pi / 2], 4)
        assert result.shape == (4, 4, 2, 2)
        z = np.zeros((4, 4), dtype=np.uint32)
        e1 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint32)
        e2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], z)
        np.testing.assert_array_equal(result[:, :, 1, 0], z)
        np.testing.assert_array_equal(result[:, :, 0, 1], e1)
        np.testing.assert_array_equal(result[:, :, 1, 1], e2)

    def test_output_empty(self):
        result = graycomatrix(self.image, [10], [0], 4)
        np.testing.assert_array_equal(result[:, :, 0, 0], np.zeros((4, 4), dtype=np.uint32))
        result = graycomatrix(self.image, [10], [0], 4, normed=True)
        np.testing.assert_array_equal(result[:, :, 0, 0], np.zeros((4, 4), dtype=np.uint32))

    def test_normed_symmetric(self):
        result = graycomatrix(self.image, [1, 2, 3], [0, np.pi / 2, np.pi], 4, normed=True, symmetric=True)
        for d in range(result.shape[2]):
            for a in range(result.shape[3]):
                np.testing.assert_almost_equal(result[:, :, d, a].sum(), 1.0)
                np.testing.assert_array_equal(result[:, :, d, a], result[:, :, d, a].transpose())

    def test_contrast(self):
        result = graycomatrix(self.image, [1, 2], [0], 4, normed=True, symmetric=True)
        result = np.round(result, 3)
        contrast = graycoprops(result, 'contrast')
        np.testing.assert_almost_equal(contrast[0, 0], 0.585, decimal=3)

    def test_dissimilarity(self):
        result = graycomatrix(self.image, [1], [0, np.pi / 2], 4, normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = graycoprops(result, 'dissimilarity')
        np.testing.assert_almost_equal(dissimilarity[0, 0], 0.418, decimal=3)

    def test_dissimilarity_2(self):
        result = graycomatrix(self.image, [1, 3], [np.pi / 2], 4, normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = graycoprops(result, 'dissimilarity')[0, 0]
        np.testing.assert_almost_equal(dissimilarity, 0.665, decimal=3)

    def test_non_normalized_glcm(self):
        img = (np.random.random((100, 100)) * 8).astype(np.uint8)
        p = graycomatrix(img, [1, 2, 4, 5], [0, 0.25, 1, 1.5], levels=8)
        np.testing.assert_(np.max(graycoprops(p, 'correlation')) < 1.0)

    def test_invalid_property(self):
        result = graycomatrix(self.image, [1], [0], 4)
        with pytest.raises(ValueError):
            graycoprops(result, 'ABC')

    def test_homogeneity(self):
        result = graycomatrix(self.image, [1], [0, 6], 4, normed=True, symmetric=True)
        homogeneity = graycoprops(result, 'homogeneity')[0, 0]
        np.testing.assert_almost_equal(homogeneity, 0.80833333)

    def test_energy(self):
        result = graycomatrix(self.image, [1], [0, 4], 4, normed=True, symmetric=True)
        energy = graycoprops(result, 'energy')[0, 0]
        np.testing.assert_almost_equal(energy, 0.38188131)

    def test_correlation(self):
        result = graycomatrix(self.image, [1, 2], [0], 4, normed=True, symmetric=True)
        energy = graycoprops(result, 'correlation')
        np.testing.assert_almost_equal(energy[0, 0], 0.71953255)
        np.testing.assert_almost_equal(energy[1, 0], 0.4117647)

    def test_uniform_properties(self):
        im = np.ones((4, 4), dtype=np.uint8)
        result = graycomatrix(im, [1, 2, 8], [0, np.pi / 2], 4, normed=True, symmetric=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            graycoprops(result, prop)