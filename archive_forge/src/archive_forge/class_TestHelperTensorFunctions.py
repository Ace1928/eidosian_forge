import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
class TestHelperTensorFunctions(unittest.TestCase):

    def test_make_string_tensor(self) -> None:
        string_list = [s.encode('utf-8') for s in ['Amy', 'Billy', 'Cindy', 'David']]
        tensor = helper.make_tensor(name='test', data_type=TensorProto.STRING, dims=(2, 2), vals=string_list, raw=False)
        self.assertEqual(string_list, list(tensor.string_data))

    def test_make_bfloat16_tensor(self) -> None:
        np_array = np.array([[1.0, 2.0], [3.0, 4.0], [0.099853515625, 0.099365234375], [0.0998535081744, 0.1], [np.nan, np.inf]], dtype=np.float32)
        np_results = np.array([[struct.unpack('!f', bytes.fromhex('3F800000'))[0], struct.unpack('!f', bytes.fromhex('40000000'))[0]], [struct.unpack('!f', bytes.fromhex('40400000'))[0], struct.unpack('!f', bytes.fromhex('40800000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCC0000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCD0000'))[0]], [struct.unpack('!f', bytes.fromhex('7FC00000'))[0], struct.unpack('!f', bytes.fromhex('7F800000'))[0]]])
        tensor = helper.make_tensor(name='test', data_type=TensorProto.BFLOAT16, dims=np_array.shape, vals=np_array)
        self.assertEqual(tensor.name, 'test')
        np.testing.assert_equal(np_results, numpy_helper.to_array(tensor))

    def test_make_float8e4m3fn_tensor(self) -> None:
        y = helper.make_tensor('zero_point', TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 50000, 10.1])
        ynp = numpy_helper.to_array(y)
        expected = np.array([0, 0.5, 1, 448, 10], dtype=np.float32)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e4m3fnuz_tensor(self) -> None:
        y = helper.make_tensor('zero_point', TensorProto.FLOAT8E4M3FNUZ, [7], [0, 0.5, 1, 50000, 10.1, -1e-05, 1e-05])
        ynp = numpy_helper.to_array(y)
        expected = np.array([0, 0.5, 1, 240, 10, 0, 0], dtype=np.float32)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e5m2_tensor(self) -> None:
        y = helper.make_tensor('zero_point', TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 50000, 96])
        ynp = numpy_helper.to_array(y)
        expected = np.array([0, 0.5, 1, 49152, 96], dtype=np.float32)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e5m2fnuz_tensor(self) -> None:
        y = helper.make_tensor('zero_point', TensorProto.FLOAT8E5M2FNUZ, [7], [0, 0.5, 1, 50000, 96, -1e-07, 1e-07])
        ynp = numpy_helper.to_array(y)
        expected = np.array([0, 0.5, 1, 49152, 96, 0, 0], dtype=np.float32)
        np.testing.assert_equal(expected, ynp)

    def test_make_bfloat16_tensor_raw(self) -> None:
        np_array = np.array([[1.0, 2.0], [3.0, 4.0], [0.099853515625, 0.099365234375], [0.0998535081744, 0.1], [np.nan, np.inf]], dtype=np.float32)
        np_results = np.array([[struct.unpack('!f', bytes.fromhex('3F800000'))[0], struct.unpack('!f', bytes.fromhex('40000000'))[0]], [struct.unpack('!f', bytes.fromhex('40400000'))[0], struct.unpack('!f', bytes.fromhex('40800000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCB0000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCC0000'))[0]], [struct.unpack('!f', bytes.fromhex('7FC00000'))[0], struct.unpack('!f', bytes.fromhex('7F800000'))[0]]])

        def truncate(x):
            return x >> 16
        values_as_ints = np_array.astype(np.float32).view(np.uint32).flatten()
        packed_values = truncate(values_as_ints).astype(np.uint16).tobytes()
        tensor = helper.make_tensor(name='test', data_type=TensorProto.BFLOAT16, dims=np_array.shape, vals=packed_values, raw=True)
        self.assertEqual(tensor.name, 'test')
        np.testing.assert_equal(np_results, numpy_helper.to_array(tensor))

    def test_make_float8e4m3fn_tensor_raw(self) -> None:
        expected = np.array([0, 0.5, 1, 448, 10], dtype=np.float32)
        f8 = np.array([helper.float32_to_float8e4m3(x) for x in expected], dtype=np.uint8)
        packed_values = f8.tobytes()
        y = helper.make_tensor(name='test', data_type=TensorProto.FLOAT8E4M3FN, dims=list(expected.shape), vals=packed_values, raw=True)
        ynp = numpy_helper.to_array(y)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e4m3fnuz_tensor_raw(self) -> None:
        expected = np.array([0, 0.5, 1, 240, 10], dtype=np.float32)
        f8 = np.array([helper.float32_to_float8e4m3(x, uz=True) for x in expected], dtype=np.uint8)
        packed_values = f8.tobytes()
        y = helper.make_tensor(name='test', data_type=TensorProto.FLOAT8E4M3FNUZ, dims=list(expected.shape), vals=packed_values, raw=True)
        ynp = numpy_helper.to_array(y)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e5m2_tensor_raw(self) -> None:
        expected = np.array([0, 0.5, 1, 49152, 10], dtype=np.float32)
        f8 = np.array([helper.float32_to_float8e5m2(x) for x in expected], dtype=np.uint8)
        packed_values = f8.tobytes()
        y = helper.make_tensor(name='test', data_type=TensorProto.FLOAT8E5M2, dims=list(expected.shape), vals=packed_values, raw=True)
        ynp = numpy_helper.to_array(y)
        np.testing.assert_equal(expected, ynp)

    def test_make_float8e5m2fnuz_tensor_raw(self) -> None:
        expected = np.array([0, 0.5, 1, 49152, 10], dtype=np.float32)
        f8 = np.array([helper.float32_to_float8e5m2(x, fn=True, uz=True) for x in expected], dtype=np.uint8)
        packed_values = f8.tobytes()
        y = helper.make_tensor(name='test', data_type=TensorProto.FLOAT8E5M2FNUZ, dims=list(expected.shape), vals=packed_values, raw=True)
        ynp = numpy_helper.to_array(y)
        np.testing.assert_equal(expected, ynp)

    @parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), ((5, 4, 6), (4, 6, 5), (3, 3), (1,), (2 ** 10,))))
    @unittest.skipIf(version_utils.numpy_older_than('1.22.0'), 'The test requires numpy 1.22.0 or later')
    def test_make_4bit_tensor(self, dtype, dims) -> None:
        type_range = {TensorProto.UINT4: (0, 15), TensorProto.INT4: (-8, 7)}
        data = np.random.randint(type_range[dtype][0], high=type_range[dtype][1] + 1, size=dims)
        y = helper.make_tensor('y', dtype, data.shape, data)
        ynp = to_array_extended(y)
        np.testing.assert_equal(data, ynp)

    @parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), ((5, 4, 6), (4, 6, 5), (3, 3), (1,))))
    def test_make_4bit_raw_tensor(self, dtype, dims) -> None:
        type_range = {TensorProto.UINT4: (0, 15), TensorProto.INT4: (-8, 7)}
        data = np.random.randint(type_range[dtype][0], high=type_range[dtype][1] + 1, size=dims)
        packed_data = helper.pack_float32_to_4bit(data, signed=dtype == TensorProto.INT4)
        y = helper.make_tensor('packed_int4', dtype, dims, packed_data.tobytes(), raw=True)
        ynp = numpy_helper.to_array(y)
        np.testing.assert_equal(data, ynp)

    def test_make_sparse_tensor(self) -> None:
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        values_tensor = helper.make_tensor(name='test', data_type=TensorProto.FLOAT, dims=(5,), vals=values)
        indices = [1, 3, 5, 7, 9]
        indices_tensor = helper.make_tensor(name='test_indices', data_type=TensorProto.INT64, dims=(5,), vals=indices)
        dense_shape = [10]
        sparse = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        self.assertEqual(sparse.values, values_tensor)
        self.assertEqual(sparse.indices, indices_tensor)
        self.assertEqual(sparse.dims, dense_shape)

    def test_make_tensor_value_info(self) -> None:
        vi = helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 4))
        checker.check_value_info(vi)
        vi = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ())
        checker.check_value_info(vi)

    def test_make_sparse_tensor_value_info(self) -> None:
        vi = helper.make_sparse_tensor_value_info('X', TensorProto.FLOAT, (2, 3))
        checker.check_value_info(vi)
        vi = helper.make_sparse_tensor_value_info('Y', TensorProto.FLOAT, ())
        checker.check_value_info(vi)