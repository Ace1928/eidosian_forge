import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
class BlockSizeExperiment(Experiment):
    """
    In this experiment, we analyze how increasing the block size affects
    performance.  We will take the lower triangular pattern. As we increase the
    batch size, the blocks near the diagonal have to do more unnecessary
    computation (the effective sparsity starts decreasing).
    """

    def __init__(self, mode, dtype, do_accuracy_check, profile_sputnik=False):
        super(BlockSizeExperiment, self).__init__(mode, dtype, do_accuracy_check, profile_sputnik)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [2048]
        block_sizes = [32, 64, 128, 256]
        hidden_sizes = [1024, 8192]
        for batch in batch_sizes:
            for seq in seq_lengths:
                for hidden_size in hidden_sizes:
                    for block in block_sizes:
                        for head in heads:
                            entry = {'batch_size': batch, 'num_heads': head, 'seq_length': seq, 'block_size': block, 'hidden_size': hidden_size}
                            yield entry

    def plot(self, sparsity, config, pattern_name):
        pretty_barplot(self.results['speedup'], title=f'{self.mode} - BlockSize experiment speedup\nbs={config.batch_size}, nheads={config.num_heads}, seq_len={config.seq_length}, dtype={self.dtype}', filename=f'vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_time.svg', dash_key='pytorch', units='Speedup normalized to torch matmul')
        pretty_barplot(self.results['flops'], title=f'{self.mode} - BlockSize experiment throughput\nbs={config.batch_size}, nheads={config.num_heads}, seq_len={config.seq_length}, dtype={self.dtype}', filename=f'vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_flops.svg', dash_key='pytorch', units='TFlops/s')
        pretty_barplot(self.results['memory_savings'], title=f'{self.mode} - BlockSize experiment memory savings\nbs={config.batch_size}, nheads={config.num_heads}, seq_len={config.seq_length}, dtype={self.dtype}', filename=f'vary_block_size_{self.mode}_{self.dtype}_{pattern_name}_memory.svg', dash_key='pytorch', units='Memory savings normalized to torch matmul')

    def get_op_flops(self, mask, config):
        num_masked_elems = (config.seq_length + 1) / (2.0 * config.seq_length)
        op_flops = 2 * config.batch_size * config.num_heads * (config.hidden_size // config.num_heads) * config.seq_length * config.seq_length * num_masked_elems * 1e-12
        return op_flops

    def run(self):
        self.reset_results()
        lt_config = None
        for config in self.gen_config():
            lt_mask, lt_config, lt_name = get_mask(LowerTriangularAttentionMask, config)
            sparsity = get_sparsity(lt_mask)
            print('Effective sparsity', sparsity)
            if lt_config.seq_length == 2048:
                plot_mask(lt_mask, lt_config, f'lt_mask_{lt_config.block_size}.svg')
            a, b = self.get_inputs(lt_config)
            tests = []
            baseline_name = 'torch-matmul'
            tests.append(TestCase(self.torch_matmul_callable, lt_mask, lt_config, f'{baseline_name}'))
            tests.append(TestCase(self.triton_callable, lt_mask, lt_config, 'triton-random'))
            if self.profile_sputnik and self.mode == 'sddmm':
                tests.append(TestCase(self.sputnik_callable, lt_mask, lt_config, 'sputnik-random'))
            dict_key = f'hidden={lt_config.hidden_size}, block={lt_config.block_size}'
            self.bench_all(a, b, tests, lt_config, sparsity, baseline_name, self.get_op_flops(lt_mask, lt_config), dict_key)
            ideal_testcase = TestCase(None, None, None, 'Ideal')
            seq_len = lt_config.seq_length
            total_elems = seq_len * seq_len
            nnz = seq_len * (seq_len + 1) / 2
            ideal_speedup = 1.0 * total_elems / nnz
            self.add_kv(self.results['speedup'], dict_key, ideal_speedup, ideal_testcase)
            self.add_kv(self.results['memory_savings'], dict_key, ideal_speedup, ideal_testcase)
        self.plot(None, lt_config, lt_name)