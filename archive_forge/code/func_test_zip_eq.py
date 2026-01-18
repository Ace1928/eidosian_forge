import pytest
import sympy
import cirq
def test_zip_eq():
    et = cirq.testing.EqualsTester()
    point_sweep1 = cirq.Points('a', [1, 2, 3])
    point_sweep2 = cirq.Points('b', [4, 5, 6, 7])
    point_sweep3 = cirq.Points('c', [1, 2])
    et.add_equality_group(cirq.ZipLongest(), cirq.ZipLongest())
    et.add_equality_group(cirq.ZipLongest(point_sweep1, point_sweep2), cirq.ZipLongest(point_sweep1, point_sweep2))
    et.add_equality_group(cirq.ZipLongest(point_sweep3, point_sweep2))
    et.add_equality_group(cirq.ZipLongest(point_sweep2, point_sweep1))
    et.add_equality_group(cirq.ZipLongest(point_sweep1, point_sweep2, point_sweep3))
    et.add_equality_group(cirq.Zip(point_sweep1, point_sweep2, point_sweep3))
    et.add_equality_group(cirq.Zip(point_sweep1, point_sweep2))