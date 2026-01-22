import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
class IterGraphModule(nn.Module):
    """``IterGraphModule`` provides the ability to do cross-iteration optimization.

    Given a ``fx.GraphModule``, main_gm, ``IterGraphModule`` internally
    duplicate it to 3 copies and redirect the ``forward`` request to a different
    ``fx.GraphModule`` based on the iteration count. This allows users to do
    graph optimizations that across iterations (e.g., moving collective wait in
    the backward to the forward of the next iteration).

    Note that users must call the APIs provided by ``IterGraphModule`` or
    ``IterGraph`` to rewrite the graph so that ``IterGraphModule`` can keep the
    data dependency for all 3 graphs.
    """

    def __init__(self, main_gm: fx.GraphModule, max_iters: int=-1, enable_inductor: bool=False) -> None:
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, 'meta', {})
            return gm
        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(main_gm, IterGraph(main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph))
        self._iter = 0
        self._max_iters = max_iters
        self._previous_output: Tuple[Any, ...] = tuple()
        self._num_extra_output = 0
        self._is_frozen = False
        self._enable_inductor = enable_inductor

    def finalize_setup(self) -> None:
        """Set up the internal states and also get the signal from users that what is the maximum iteration count.

        This method must be called before the forward() is called.
        """
        if not self._is_frozen:
            self.graph.freeze_cross_iter_movement()
            self._num_extra_output = self.graph.num_extra_output
            if self._enable_inductor:
                self.main_gm = partial_lower(self.main_gm)
            self._is_frozen = True
        self._iter = 0

    def _run(self, gm: fx.GraphModule, last_iter: bool, *args, **kwargs) -> Any:
        if self._num_extra_output > 0:
            new_args = args + self._previous_output
            output = gm(*new_args, **kwargs)
            if not last_iter:
                assert len(output) == 2
                self._previous_output = tuple(output[-1])
                assert len(self._previous_output) > 0, 'There should be at least one extra output.'
                output = output[0]
        else:
            output = gm(*args, **kwargs)
        return output

    def forward(self, *args: Any, last_iter: bool=False, **kwargs: Any) -> Any:
        self._iter += 1
        last_iter = last_iter or self._iter == self._max_iters
        if last_iter:
            logger.info('Using the cleanup graph')
            gm = self.cleanup_gm
            profiler_string = '## IterGraphModule: Cleanup Graph ##'
            self._iter = 0
        elif self._iter == 1:
            logger.info('Using the setup graph')
            gm = self.setup_gm
            profiler_string = '## IterGraphModule: Setup Graph ##'
        else:
            gm = self.main_gm
            if self._iter == 2:
                logger.info('Using the main graph')
                profiler_string = '## IterGraphModule -- Maybe Compiling ##'
            else:
                profiler_string = '## IterGraphModule ##'
        with record_function(profiler_string):
            return self._run(gm, last_iter, *args, **kwargs)

    @property
    def graph(self) -> IterGraph:
        return cast(IterGraph, self.main_gm.graph)

    def recompile(self) -> PythonCode:
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()

    def freeze_cross_iter_movement(self) -> None:
        self.graph.freeze_cross_iter_movement()
        self._num_extra_output = self.graph.num_extra_output

    def print_readable(self, print_output: bool=True) -> str:
        return self.main_gm.print_readable(print_output)

    def print_all_graphs(self) -> None:
        logger.info('Printing the three fx.Graph:')
        logger.info('1. Setup fx.Graph:')
        logger.info('%s', self.setup_gm.graph)
        logger.info('2. Main fx.Graph:')
        logger.info('%s', self.main_gm.graph)
        logger.info('3. Cleanup fx.Graph:')
        logger.info('%s', self.cleanup_gm.graph)

    def print_all_graph_modules(self) -> None:
        logger.info('Printing the three fx gm:')
        logger.info('1. Setup fx.GraphModule:')
        logger.info('%s', self.setup_gm.print_readable(False))
        logger.info('2. Main fx.GraphModule:')
        logger.info('%s', self.main_gm.print_readable(False))
        logger.info('3. Cleanup fx.GraphModule:')
        logger.info('%s', self.cleanup_gm.print_readable(False))