import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def _test_merge_models(self, m1def: str, m2def: str, io_map: List[Tuple[str, str]], check_expectations: Callable[[GraphProto, GraphProto, GraphProto], None], inputs: Optional[List[str]]=None, outputs: Optional[List[str]]=None, prefix1: Optional[str]=None, prefix2: Optional[str]=None) -> None:
    m1, m2 = (_load_model(m1def), _load_model(m2def))
    g3 = compose.merge_graphs(m1.graph, m2.graph, io_map=io_map, inputs=inputs, outputs=outputs, prefix1=prefix1, prefix2=prefix2)
    checker.check_graph(g3)
    check_expectations(m1.graph, m2.graph, g3)
    m3 = compose.merge_models(m1, m2, io_map=io_map, inputs=inputs, outputs=outputs, prefix1=prefix1, prefix2=prefix2)
    checker.check_model(m3)
    check_expectations(m1.graph, m2.graph, m3.graph)