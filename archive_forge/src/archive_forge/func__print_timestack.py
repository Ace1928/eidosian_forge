import timeit
import math
def _print_timestack(stack, level=1):
    print('-' * level, '%.2f %s%s' % (stack[2], stack[0], stack[3]))
    for s in stack[1]:
        _print_timestack(s, level + 1)