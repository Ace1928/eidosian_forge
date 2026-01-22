import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
class CompileProfiler:
    """Utility for profiling how and what dynamo would compile.

    Can be used for
     * diagnosing recompilation issues
     * determining an appropriate compile cache limit
     * (TODO)confirming which functions got compiled/skipped
    """

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.backend_ctx_ctor = disable_cache_limit

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if 'call' in node.op:
                self.op_count += 1
        return gm.forward

    def __enter__(self):
        return self

    def __exit__(self, typ, val, traceback):
        pass

    def get_metrics(self):
        return {'guard_failures': guard_failures}

    def report(self):
        metrics = self.get_metrics()
        gf = metrics['guard_failures']

        def num_recompiles(code):
            return len(gf[code])

        def recompile_reasons(code):
            return '\n'.join([str(x) for x in gf[code]])
        summarized_gf = [[format_func_info(code), num_recompiles(code), recompile_reasons(code)] for code in gf]

        def graph_break_report():
            if 'graph_break' in counters:
                graph_breaks = counters['graph_break']
                return tabulate([[msg, graph_breaks[msg]] for msg in graph_breaks], headers=['Graph Break Reason', 'Count'])

        def recompilation_report():
            if len(gf):
                max_recompiles = max([num_recompiles(code) for code in gf])
                recomp_table = tabulate(summarized_gf, headers=['Function', 'Recompiles', 'Recompile Reasons'])
                return recomp_table + textwrap.dedent(f'\n\n                    Set torch._dynamo.config.cache_size_limit to {max_recompiles} to avoid being cache limited.\n                ')
        report = textwrap.dedent("\n            Torchdynamo Profiler Report\n            ===========================\n\n            Graph Breaks\n            ------------\n            Graph breaks happen when torchdynamo encounters code it can't safely trace.\n            If you want to find out why breaks are happening, check below for each break reason\n            You may gain additional insight by passing `fullgraph=True` to torch.compile,\n            to stop at the first break.\n\n        ")
        report += graph_break_report() or 'No graph breaks detected.'
        report += textwrap.dedent('\n\n            Recompilation\n            -------------\n            These subgraphs were recompiled more than once due to guard failures\n            Guard failures indicate some condition assumed to be static by the tracer changed,\n            making it unsafe to reuse the compiled program.\n\n        ')
        report += recompilation_report() or 'No recompilation detected.\n'
        return report