from ..pari import pari
import fractions
def _debug_print_matrix(m):
    for row in m:
        print('    ', end=' ')
        for c in row:
            print('%4d' % c, end=' ')
        print()
    print()
    print()