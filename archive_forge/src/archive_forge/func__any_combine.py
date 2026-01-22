import triton
import triton.language as tl
@triton.jit
def _any_combine(a, b):
    return a | b