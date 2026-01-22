import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _single_qubit_cliffords() -> Cliffords:
    X, Y, Z = (ops.X, ops.Y, ops.Z)
    c1_in_xy: List[List['cirq.Gate']] = []
    c1_in_xz: List[List['cirq.Gate']] = []
    for phi_0, phi_1 in itertools.product([1.0, 0.5, -0.5], [0.0, 0.5, -0.5]):
        c1_in_xy.append([X ** phi_0, Y ** phi_1])
        c1_in_xy.append([Y ** phi_0, X ** phi_1])
        c1_in_xz.append([X ** phi_0, Z ** phi_1])
        c1_in_xz.append([Z ** phi_0, X ** phi_1])
    c1_in_xy.append([X ** 0.0])
    c1_in_xz.append([X ** 0.0])
    c1_in_xy.append([Y, X])
    c1_in_xz.append([Z, X])
    phi_xy = [[-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]
    for y0, x, y1 in phi_xy:
        c1_in_xy.append([Y ** y0, X ** x, Y ** y1])
    phi_xz = [[0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5]]
    for z0, x, z1 in phi_xz:
        c1_in_xz.append([Z ** z0, X ** x, Z ** z1])
    s1: List[List['cirq.Gate']] = [[X ** 0.0], [Y ** 0.5, X ** 0.5], [X ** (-0.5), Y ** (-0.5)]]
    s1_x: List[List['cirq.Gate']] = [[X ** 0.5], [X ** 0.5, Y ** 0.5, X ** 0.5], [Y ** (-0.5)]]
    s1_y: List[List['cirq.Gate']] = [[Y ** 0.5], [X ** (-0.5), Y ** (-0.5), X ** 0.5], [Y, X ** 0.5]]
    return Cliffords(c1_in_xy, c1_in_xz, s1, s1_x, s1_y)