from mpmath import zetazero
from timeit import default_timer as clock
def test_zetazero():
    cases = [(399999999, 156762524.67505914), (241389216, 97490234.22767118), (526196239, 202950727.69122952), (542964976, 209039046.57853526), (1048449112, 388858885.2310565), (1048449113, 388858885.3843374), (1048449114, 388858886.0022851), (1048449115, 388858886.00239366), (1048449116, 388858886.69074506)]
    for n, v in cases:
        print(n, v)
        t1 = clock()
        ok = zetazero(n).ae(complex(0.5, v))
        t2 = clock()
        print('ok =', ok, '(time = %s)' % round(t2 - t1, 3))
    print('Now computing two huge zeros (this may take hours)')
    print('Computing zetazero(8637740722917)')
    ok = zetazero(8637740722917).ae(complex(0.5, 2124447368584.393))
    print('ok =', ok)
    ok = zetazero(8637740722918).ae(complex(0.5, 2124447368584.393))
    print('ok =', ok)