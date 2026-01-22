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
class DifferentPatternExperiment(Experiment):
    """
    In this experiment, we check if sparsity pattern (like bigbird, lower triangular
    etc) changes the performance of different kernels. The idea is to check if
    changing sparsity pattern, while keeping total sparsity ratio same, leads to any
    perforamnce differences.

    We will perform two experiments

    1) LowerTraingularMask vs RandomMask - Both have ~50% sparsity.
    2) BigBird Mask vs RandomMask - Both have same sparsity.
    """

    def __init__(self, mode, dtype, do_accuracy_check, profile_sputnik=False):
        super(DifferentPatternExperiment, self).__init__(mode, dtype, do_accuracy_check, profile_sputnik)

    def gen_config(self):
        batch_sizes = [32]
        heads = [16]
        seq_lengths = [1024, 2048]
        block_sizes = [64]
        hidden_sizes = [1024, 4096, 8192]
        for batch in batch_sizes:
            for hidden_size in hidden_sizes:
                for head in heads:
                    for seq in seq_lengths:
                        for block in block_sizes:
                            entry = {'batch_size': batch, 'num_heads': head, 'seq_length': seq, 'block_size': block, 'hidden_size': hidden_size}
                            yield entry

    def plot(self, sparsity, config, pattern_name):
        desc = [f'bs={config.batch_size}', f'nheads={config.num_heads}', f'block={config.block_size}', f'dtype={self.dtype}']
        title_suffix = ','.join(desc)
        pretty_barplot(self.results['speedup'], title=f'{self.mode} - Pattern experiment ({sparsity}%) - speedup\n' + title_suffix, filename=f'same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_time.svg', dash_key='pytorch', units='Speedup normalized to torch_matmul')
        pretty_barplot(self.results['flops'], title=f'{self.mode} - Pattern experiment ({sparsity}%) - throughput\n' + title_suffix, filename=f'same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_flops.svg', dash_key='pytorch', units='TFlops/s')
        pretty_barplot(self.results['memory_savings'], title=f'{self.mode} - Pattern experiment ({sparsity}%) - memory savings\n' + title_suffix, filename=f'same_sparsity_{self.mode}_{self.dtype}_{pattern_name}_memory.svg', dash_key='pytorch', units='Memory savings normalized to torch_matmul')

    def run(self):
        for MaskGenType in [LowerTriangularAttentionMask, BigBirdAttentionMask]:
            self.reset_results()
            for config in self.gen_config():
                pattern_mask, pattern_config, pattern_name = get_mask(MaskGenType, config)
                sparsity = get_sparsity(pattern_mask)
                mask_prob = sparsity / 100
                random_mask, random_config, _ = get_mask(RandomAttentionMask, config, [('mask_prob', mask_prob)])
                print(f'{pattern_name} sparsity', get_sparsity(pattern_mask))
                print('Random sparsity', get_sparsity(random_mask))
                a, b = self.get_inputs(random_config)
                tests = []
                baseline_name = 'torch-matmul'
                tests.append(TestCase(self.torch_matmul_callable, random_mask, random_config, f'{baseline_name}'))
                tests.append(TestCase(self.triton_callable, random_mask, random_config, 'triton-random'))
                tests.append(TestCase(self.triton_callable, pattern_mask, pattern_config, f'triton-{pattern_name}'))
                if self.profile_sputnik and self.mode == 'sddmm':
                    tests.append(TestCase(self.sputnik_callable, random_mask, random_config, 'sputnik-random'))
                    tests.append(TestCase(self.sputnik_callable, pattern_mask, pattern_config, f'sputnik-{pattern_name}'))
                dict_key = f'hidden={random_config.hidden_size},seq_len={random_config.seq_length}'
                self.bench_all(a, b, tests, random_config, sparsity, baseline_name, self.get_op_flops(random_mask, random_config), dict_key)
                ideal_testcase = TestCase(None, None, None, 'Ideal')
                ideal_speedup = round(100 / (100 - sparsity), 1)
                self.add_kv(self.results['speedup'], dict_key, ideal_speedup, ideal_testcase)
                self.add_kv(self.results['memory_savings'], dict_key, ideal_speedup, ideal_testcase)
                self.plot(sparsity, random_config, pattern_name)