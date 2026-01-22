import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
class DataParallelExecutorManager(object):
    """ Helper class to manage multiple executors for data parallelism.

    Parameters
    ----------
    symbol : Symbol
        Output symbol.
    ctx : list of Context
        Devices to run on.
    param_names: list of str
        Name of all trainable parameters of the network.
    arg_names: list of str
        Name of all arguments of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    train_data : DataIter
        Training data iterator.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx.
    logger : logging logger
        When not specified, default logger will be used.
    sym_gen : A function that generate new Symbols depending on different
        input shapes. Used only for bucketing.
    """

    def __init__(self, symbol, ctx, train_data, arg_names, param_names, aux_names, work_load_list=None, logger=None, sym_gen=None):
        if logger is None:
            logger = logging
        num_device = len(ctx)
        logger.info('Start training with %s', str(ctx))
        if work_load_list is None:
            work_load_list = [1] * num_device
        assert isinstance(work_load_list, list) and len(work_load_list) == num_device, 'Invalid settings for work load. '
        slices = _split_input_slice(train_data.batch_size, work_load_list)
        self.slices = slices
        self.arg_names = arg_names
        self.param_names = param_names
        self.aux_names = aux_names
        self.ctx = ctx
        self.execgrp = DataParallelExecutorGroup(symbol, self.arg_names, self.param_names, self.ctx, self.slices, train_data)
        self.symbol = symbol
        self.sym_gen = sym_gen
        self.curr_execgrp = None
        if self.sym_gen is not None:
            self.execgrp_bucket = {train_data.default_bucket_key: self.execgrp}

    def install_monitor(self, monitor):
        """Install monitor on all executors."""
        if self.sym_gen is not None:
            raise NotImplementedError('Monitoring is not implemented for bucketing')
        for train_exec in self.execgrp.train_execs:
            monitor.install(train_exec)

    def set_params(self, arg_params, aux_params):
        """Set parameter and aux values.

        Parameters
        ----------
        arg_params : list of NDArray
            Source parameter arrays
        aux_params : list of NDArray
            Source aux arrays.
        """
        for texec in self.execgrp.train_execs:
            texec.copy_params_from(arg_params, aux_params)

    def copy_to(self, arg_params, aux_params):
        """ Copy data from each executor to ```arg_params`` and ``aux_params``.

        Parameters
        ----------
        arg_params : list of NDArray
            Target parameter arrays.
        aux_params : list of NDArray
            Target aux arrays.

        Notes
        -----
        - This function will inplace update the NDArrays in arg_params and aux_params.
        """
        for name, block in zip(self.param_names, self.param_arrays):
            weight = sum((w.copyto(cpu()) for w in block)) / len(block)
            weight.astype(arg_params[name].dtype).copyto(arg_params[name])
        for name, block in zip(self.aux_names, self.aux_arrays):
            weight = sum((w.copyto(cpu()) for w in block)) / len(block)
            weight.astype(aux_params[name].dtype).copyto(aux_params[name])

    @property
    def param_arrays(self):
        """Shared parameter arrays."""
        return self.execgrp.param_arrays

    @property
    def grad_arrays(self):
        """Shared gradient arrays."""
        return self.execgrp.grad_arrays

    @property
    def aux_arrays(self):
        """Shared aux states."""
        return self.execgrp.aux_arrays

    def load_data_batch(self, data_batch):
        """Load data and labels into arrays."""
        if self.sym_gen is not None:
            key = data_batch.bucket_key
            if key not in self.execgrp_bucket:
                symbol = self.sym_gen(key)
                execgrp = DataParallelExecutorGroup(symbol, self.arg_names, self.param_names, self.ctx, self.slices, data_batch, shared_group=self.execgrp)
                self.execgrp_bucket[key] = execgrp
            self.curr_execgrp = self.execgrp_bucket[key]
        else:
            self.curr_execgrp = self.execgrp
        self.curr_execgrp.load_data_batch(data_batch)

    def forward(self, is_train=False):
        """Run forward on the current executor."""
        self.curr_execgrp.forward(is_train=is_train)

    def backward(self):
        """Run backward on the current executor."""
        self.curr_execgrp.backward()

    def update_metric(self, metric, labels, pre_sliced=False):
        """Update metric with the current executor."""
        self.curr_execgrp.update_metric(metric, labels, pre_sliced)