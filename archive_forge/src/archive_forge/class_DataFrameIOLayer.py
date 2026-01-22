from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
class DataFrameIOLayer(Blockwise):
    """DataFrame-based Blockwise Layer with IO

    Parameters
    ----------
    name : str
        Name to use for the constructed layer.
    columns : str, list or None
        Field name(s) to read in as columns in the output.
    inputs : list or BlockwiseDep
        List of arguments to be passed to ``io_func`` so
        that the materialized task to produce partition ``i``
        will be: ``(<io_func>, inputs[i])``.  Note that each
        element of ``inputs`` is typically a tuple of arguments.
    io_func : callable
        A callable function that takes in a single tuple
        of arguments, and outputs a DataFrame partition.
        Column projection will be supported for functions
        that satisfy the ``DataFrameIOFunction`` protocol.
    label : str (optional)
        String to use as a prefix in the place-holder collection
        name. If nothing is specified (default), "subset-" will
        be used.
    produces_tasks : bool (optional)
        Whether one or more elements of `inputs` is expected to
        contain a nested task. This argument in only used for
        serialization purposes, and will be deprecated in the
        future. Default is False.
    creation_info: dict (optional)
        Dictionary containing the callable function ('func'),
        positional arguments ('args'), and key-word arguments
        ('kwargs') used to produce the dask collection with
        this underlying ``DataFrameIOLayer``.
    annotations: dict (optional)
        Layer annotations to pass through to Blockwise.
    """

    def __init__(self, name, columns, inputs, io_func, label=None, produces_tasks=False, creation_info=None, annotations=None):
        self.name = name
        self._columns = columns
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info
        if not isinstance(inputs, BlockwiseDep):
            io_arg_map = BlockwiseDepDict({(i,): inp for i, inp in enumerate(self.inputs)}, produces_tasks=self.produces_tasks)
        else:
            io_arg_map = inputs
        dsk = {self.name: (io_func, blockwise_token(0))}
        super().__init__(output=self.name, output_indices='i', dsk=dsk, indices=[(io_arg_map, 'i')], numblocks={}, annotations=annotations)

    @property
    def columns(self):
        """Current column projection for this layer"""
        return self._columns

    def project_columns(self, columns):
        """Produce a column projection for this IO layer.
        Given a list of required output columns, this method
        returns the projected layer.
        """
        from dask.dataframe.io.utils import DataFrameIOFunction
        columns = list(columns)
        if self.columns is None or set(self.columns).issuperset(columns):
            if isinstance(self.io_func, DataFrameIOFunction):
                io_func = self.io_func.project_columns(columns)
            else:
                io_func = self.io_func
            layer = DataFrameIOLayer((self.label or 'subset') + '-' + tokenize(self.name, columns), columns, self.inputs, io_func, label=self.label, produces_tasks=self.produces_tasks, annotations=self.annotations)
            return layer
        else:
            return self

    def __repr__(self):
        return "DataFrameIOLayer<name='{}', n_parts={}, columns={}>".format(self.name, len(self.inputs), self.columns)