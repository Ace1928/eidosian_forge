import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from importlib import import_module
from typing import Union
import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm
from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
def wrap_backend_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)
        assert config.repro_after in ('dynamo', 'aot', None)
        if config.repro_after == 'dynamo':

            def add_paths(exc):
                exc.minifier_path = os.path.join(minifier_dir(), 'minifier_launcher.py')
                if use_buck:
                    exc.buck_command = ' '.join(BUCK_CMD_PREFIX + [BuckTargetWriter(exc.minifier_path).cmd_line_path])
            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, compiler_name)
            if config.repro_level == 4:
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if backend_accuracy_fails(gm, example_inputs, compiler_fn):
                    log.warning('Accuracy failed for the TorchDynamo produced graph. Creating script to minify the error.')
                    dump_to_minify_after_dynamo(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs, compiler_name)
                    exc = AccuracyError('Bad accuracy detected.')
                    add_paths(exc)
                    raise exc
            else:
                try:
                    compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                    run_fwd_maybe_bwd(compiled_gm, example_inputs)
                except Exception as exc:
                    log.warning('Compiled Fx GraphModule failed. Creating script to minify the error.')
                    if config.repro_level == 1:
                        dump_state_fn = functools.partial(dump_backend_state, compiler_name=compiler_name)
                        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
                    elif config.repro_level == 2:
                        dump_to_minify_after_dynamo(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs, compiler_name)
                    add_paths(exc)
                    raise
        else:
            compiled_gm = compiler_fn(gm, example_inputs)
        return compiled_gm
    debug_wrapper._torchdynamo_orig_callable = unconfigured_compiler_fn
    if hasattr(unconfigured_compiler_fn, 'compiler_name'):
        debug_wrapper.__name__ = unconfigured_compiler_fn.compiler_name
    if hasattr(unconfigured_compiler_fn, 'get_compiler_config'):
        debug_wrapper.get_compiler_config = unconfigured_compiler_fn.get_compiler_config
    return debug_wrapper