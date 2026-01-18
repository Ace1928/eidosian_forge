import numpy as np
from ..rcparams import rcParams
from .plot_utils import get_plotting_function
def wilkinson_algorithm(values, binwidth):
    """Wilkinson's algorithm to distribute dots into horizontal stacks."""
    ndots = len(values)
    count = 0
    stack_locs, stack_counts = ([], [])
    while count < ndots:
        stack_first_dot = values[count]
        num_dots_stack = 0
        while values[count] < binwidth + stack_first_dot:
            num_dots_stack += 1
            count += 1
            if count == ndots:
                break
        stack_locs.append((stack_first_dot + values[count - 1]) / 2)
        stack_counts.append(num_dots_stack)
    return (stack_locs, stack_counts)