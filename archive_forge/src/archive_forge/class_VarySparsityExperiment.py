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
class VarySparsityExperiment(Experiment):
    """
    In this experiment, we check how sparsity ration affects the performance.
    """

    def __init__(self, mode, dtype, do_accuracy_check, profile_sputnik=False):
        super(VarySparsityExperiment, self).__init__(mode, dtype, do_accuracy_check, profile_sputnik)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [2048]
        hidden_sizes = [1024, 8192]
        block_sizes = [64]
        for batch in batch_sizes:
            for seq in seq_lengths:
                for head in heads:
                    for block in block_sizes:
                        for hidden_size in hidden_sizes:
                            entry = {'batch_size': batch, 'num_heads': head, 'seq_length': seq, 'block_size': block, 'hidden_size': hidden_size}
                            yield entry

    def plot(self, sparsity, config, pattern_name):
        desc = [f'bs={config.batch_size}', f'nheads={config.num_heads}', f'block={config.block_size}', f'dtype={self.dtype}', f'seq_len={config.seq_length}']
        title_suffix = ','.join(desc)
        pretty_barplot(self.results['speedup'], title=f'{self.mode} - SparsityRatio experiment speedup\n' + title_suffix, filename=f'vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_time.svg', dash_key='pytorch', units='Speedup normalized to torch_matmul')
        pretty_barplot(self.results['flops'], title=f'{self.mode} - SparsityRatio experiment throughput\n' + title_suffix, filename=f'vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_flops.svg', dash_key='pytorch', units='TFlops/s')
        pretty_barplot(self.results['memory_savings'], title=f'{self.mode} - SparsityRatio experiment memory savings\n' + title_suffix, filename=f'vary_sparsity_{self.mode}_{self.dtype}_{pattern_name}_memory.svg', dash_key='pytorch', units='Memory savings normalized to torch_matmul')

    def run(self):
        self.reset_results()
        random_config = None
        for config in self.gen_config():
            for x in range(10, 100, 10):
                mask_prob = x / 100.0
                random_mask, random_config, _ = get_mask(RandomAttentionMask, config, [('mask_prob', mask_prob)])
                sparsity = get_sparsity(random_mask)
                print('Random sparsity', get_sparsity(random_mask))
                a, b = self.get_inputs(random_config)
                tests = []
                baseline_name = 'torch-matmul'
                tests.append(TestCase(self.torch_matmul_callable, random_mask, random_config, f'{baseline_name}'))
                tests.append(TestCase(self.triton_callable, random_mask, random_config, 'triton-random'))
                if self.profile_sputnik and self.mode == 'sddmm':
                    tests.append(TestCase(self.sputnik_callable, random_mask, random_config, 'sputnik-random'))
                dict_key = f'sp={mask_prob},hidden={random_config.hidden_size}'
                self.bench_all(a, b, tests, random_config, sparsity, baseline_name, self.get_op_flops(random_mask, random_config), dict_key)
                ideal_testcase = TestCase(None, None, None, 'Ideal')
                ideal_speedup = round(100 / (100 - mask_prob * 100), 1)
                self.add_kv(self.results['speedup'], dict_key, ideal_speedup, ideal_testcase)
                self.add_kv(self.results['memory_savings'], dict_key, ideal_speedup, ideal_testcase)
        self.plot(None, random_config, 'random')