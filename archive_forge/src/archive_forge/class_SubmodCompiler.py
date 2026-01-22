import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
class SubmodCompiler(torch.fx.interpreter.Interpreter):

    def __init__(self, module, compiler):
        super().__init__(module)
        self.compiler = compiler

    def compile_submod(self, input_mod, args, kwargs):
        """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
        assert len(kwargs) == 0, 'We assume only args for these modules'

        class WrapperModule(torch.nn.Module):

            def __init__(self, submod, unwrap_singleton_tuple):
                super().__init__()
                self.submod = submod
                self.unwrap_singleton_tuple = unwrap_singleton_tuple

            def forward(self, *args):
                x = self.submod(*args)
                if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                    return x[0]
                return x
        unwrap_singleton_tuple = False
        for sn in input_mod.graph.nodes:
            if sn.op == 'output':
                if not isinstance(sn.args[0], tuple):
                    unwrap_singleton_tuple = True
                    sn.args = (sn.args,)
        input_mod.recompile()
        input_mod.compile_subgraph_reason = GraphCompileReason('DDPOptimizer intentional graph-break (See Note [DDPOptimizer]). Set `torch._dynamo.config.optimize_ddp = False` to disable.', [traceback.FrameSummary(__file__, 0, DDPOptimizer)])
        wrapper = WrapperModule(self.compiler(input_mod, args), unwrap_singleton_tuple)
        return wrapper

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        new_args = []
        assert fake_mode
        for arg in args:
            if isinstance(arg, torch.Tensor) and (not isinstance(arg, torch._subclasses.FakeTensor)):
                new_args.append(torch._dynamo.utils.to_fake_tensor(arg, fake_mode))
            else:
                new_args.append(arg)
        log.debug('run_node %s, %s got args %s', n.op, n.target, args_str(args))
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        if n.op == 'call_module':
            real_mod = self.fetch_attr(n.target)
            if fake_mode:
                curr_submod = deepcopy_to_fake_tensor(real_mod, fake_mode)
            else:
                curr_submod = real_mod
            ddp_graph_log.debug('\n---%s graph---\n%s', n.target, curr_submod.graph)
            compiled_submod_real = self.compile_submod(real_mod, new_args, kwargs)
            self.module.delete_submodule(n.target)
            n.target = 'compiled_' + n.target
            self.module.add_submodule(n.target, compiled_submod_real)
            with fake_mode:
                return curr_submod(*new_args, **kwargs)
        else:
            return getattr(self, n.op)(n.target, new_args, kwargs)