import triton
import triton.language as tl
@triton.jit
def welford_reduce(value, mean, m2, weight):
    delta = value - mean
    new_weight = weight + 1
    new_mean = mean + delta / new_weight
    return (new_mean, m2 + delta * (value - new_mean), new_weight)