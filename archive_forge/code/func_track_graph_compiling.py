import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
@contextmanager
def track_graph_compiling(aot_config, graph_name):
    global graph_being_compiled
    graph_being_compiled = [f'{aot_config.aot_id}_{graph_name}']
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []