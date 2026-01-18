from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
def test_random_layout(nodes_without_positions, edges):
    expected_x = [0.417022004703, 0.000114374817, 0.146755890817, 0.186260211378, 0.396767474231]
    expected_y = [0.720324493442, 0.302332572632, 0.092338594769, 0.345560727043, 0.538816734003]
    df = random_layout(nodes_without_positions, edges, seed=1)
    assert np.allclose(df['x'], expected_x)
    assert np.allclose(df['y'], expected_y)