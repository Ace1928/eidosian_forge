import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
class AlgorithmSelectorCache(PersistentCache):

    def __call__(self, name, choices: List[ChoiceCaller], input_nodes, layout, input_gen_fns: Optional[Dict[int, Callable[[ir.Buffer], torch.Tensor]]]=None):
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller
        choices = [choice for choice in choices if choice is not None]
        if len(choices) == 0:
            raise RuntimeError('No choices to select, please consider adding ATEN into max_autotune_gemm_backends config (defined in torch/_inductor/config.py) to allow at least one choice. ')
        log.info('Max autotune selects from %s choices.', str(len(choices)))
        if len(choices) == 1:
            if not isinstance(choices[0], CUDATemplateCaller):
                return choices[0].output_node()

        @functools.lru_cache(None)
        def make_benchmark_fn():
            return self.make_benchmark_fn(choices, input_nodes, layout, input_gen_fns)

        def autotune(choices):
            return make_benchmark_fn()(choices)
        if config.autotune_in_subproc:
            from .autotune_process import tuning_pool
            tuning_pool.initialize()
        autotune_start_ts = time.time()
        timings = self.lookup(choices, name, repr([self.key_of(x) for x in input_nodes]), autotune)
        autotune_elapse = time.time() - autotune_start_ts
        if timings == {} or choices[0] not in timings:
            return choices[0].output_node()
        if make_benchmark_fn.cache_info().currsize:
            counters['inductor']['select_algorithm_autotune'] += 1
        if make_benchmark_fn.cache_info().currsize or log.getEffectiveLevel() == logging.DEBUG:
            self.log_results(name, input_nodes, timings, autotune_elapse)
        selected_choice = builtins.min(timings, key=timings.__getitem__).output_node()
        log.debug('selected choice: %s', str(selected_choice))
        return selected_choice

    @classmethod
    def make_benchmark_fn(cls, choices, input_nodes, layout, input_gen_fns=None):
        if input_gen_fns is None:
            input_gen_fns = {}
        unique_example_inputs = {x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x) for i, x in enumerate(input_nodes)}
        example_inputs = list(unique_example_inputs.values())
        example_inputs_extern = [torch.as_strided(unique_example_inputs[input_node.get_name()], V.graph.sizevars.size_hints(input_node.get_size(), fallback=config.unbacked_symint_fallback), V.graph.sizevars.size_hints(input_node.get_stride(), fallback=config.unbacked_symint_fallback), V.graph.sizevars.size_hint(input_node.get_layout().offset, fallback=config.unbacked_symint_fallback)) for input_node in input_nodes]
        out = cls.benchmark_example_value(layout)
        out_extern = torch.as_strided(out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset))
        if VERIFY:
            choices[0].benchmark(*example_inputs_extern, out=out_extern)
            expected = out_extern.clone()
        if DEBUG:
            print(f'{len(choices)} tuning requests:')

        def debug_str():

            def tensor_repr(x):
                return f'torch.empty_strided({tuple(x.size())!r}, {tuple(x.stride())!r}, dtype={x.dtype!r}, device={x.device.type!r})'
            lines = ['inputs = [']
            for x in example_inputs:
                lines.append(f'    {tensor_repr(x)},')
            lines += [']', f'out = {tensor_repr(out)}', '']
            return '\n'.join(lines)

        def benchmark_choice_in_current_process(choice):
            out.zero_()
            if isinstance(choice, ExternKernelCaller):
                result = choice.benchmark(*example_inputs_extern, out=out_extern)
            else:
                result = choice.benchmark(*example_inputs, out=out)
            if VERIFY:
                torch.testing.assert_close(out_extern, expected, **VERIFY)
            torch.cuda.synchronize()
            return result

        def benchmark_in_current_process(choices):
            timings = {}
            for choice in choices:
                try:
                    timing = benchmark_choice_in_current_process(choice)
                except CUDACompileError as e:
                    log.warning('CUDA compilation error: \n%s. \nIgnore this choice.', str(e))
                    timing = float('inf')
                except RuntimeError as e:
                    msg = str(e)
                    if 'invalid argument' in msg:
                        msg += '\n\nThis may mean this GPU is too small for max_autotune mode.\n\n'
                        log.warning(msg)
                        timing = float('inf')
                    else:
                        if 'illegal memory access' in msg:
                            msg += '\n\nEither error in template or triton bug.\n'
                        raise ErrorFromChoice(msg, choice, debug_str())
                except AssertionError as e:
                    raise AssertionError(f'Incorrect result from choice {choice}\n\n{e}')
                timings[choice] = timing
            return timings

        def benchmark_in_sub_process(choices):
            from . import autotune_process
            extern = [c for c in choices if isinstance(c, ExternKernelCaller)]
            triton = [c for c in choices if not isinstance(c, ExternKernelCaller)]
            timings = benchmark_in_current_process(extern)
            timings.update(autotune_process.benchmark_in_sub_process(triton))
            return timings
        benchmark = benchmark_in_sub_process if config.autotune_in_subproc else benchmark_in_current_process
        return benchmark

    @staticmethod
    def log_results(name, input_nodes, timings, elapse):
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        sizes = ', '.join(['x'.join(map(str, V.graph.sizevars.size_hints(n.get_size(), fallback=config.unbacked_symint_fallback))) for n in input_nodes])
        n = None if log.getEffectiveLevel() == logging.DEBUG else 10
        top_k = sorted(timings, key=timings.__getitem__)[:n]
        best = top_k[0]
        best_time = timings[best]
        sys.stderr.write(f'AUTOTUNE {name}({sizes})\n')
        for choice in top_k:
            result = timings[choice]
            if result:
                sys.stderr.write(f'  {choice.name} {result:.4f} ms {best_time / result:.1%}\n')
            else:
                sys.stderr.write(f'  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>\n')
        autotune_type_str = 'SubProcess' if config.autotune_in_subproc else 'SingleProcess'
        sys.stderr.write(f'{autotune_type_str} AUTOTUNE takes {elapse:.4f} seconds\n')

    @staticmethod
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        if isinstance(node, ir.Layout):
            node = ir.Buffer('fake', node)
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()
        with preserve_rng_state():
            return rand_strided(V.graph.sizevars.size_hints(node.get_size(), fallback=config.unbacked_symint_fallback), V.graph.sizevars.size_hints(node.get_stride(), fallback=config.unbacked_symint_fallback), device=node.get_device(), dtype=node.get_dtype(), extra_size=node.layout.offset)

    @staticmethod
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
        sizevars = V.graph.sizevars
        return (node.get_device().type, str(node.get_dtype()), *sizevars.size_hints(node.get_size(), fallback=config.unbacked_symint_fallback), *sizevars.size_hints(node.get_stride(), fallback=config.unbacked_symint_fallback), sizevars.size_hint(node.get_layout().offset, fallback=config.unbacked_symint_fallback))