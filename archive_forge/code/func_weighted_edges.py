from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datashader.bundling import directly_connect_edges, hammer_bundle
from datashader.layout import circular_layout, forceatlas2_layout, random_layout
@pytest.fixture
def weighted_edges():
    edges_df = pd.DataFrame({'id': np.arange(4), 'source': np.zeros(4, dtype=int), 'target': np.arange(1, 5), 'weight': np.ones(4)})
    edges_df.set_index('id')
    return edges_df