import torch
from xformers.benchmarks.utils import TestCase, bench_functions
from xformers.triton.softmax import log_softmax as triton_log_softmax
from xformers.triton.softmax import softmax as triton_softmax
def triton_causal_fw(x):
    _ = triton_softmax(x, causal=True)