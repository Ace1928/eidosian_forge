def testparse(s, dump=0):
    from time import time
    from pprint import pprint
    now = time()
    D = parsexmlSimple(s, oneOutermostTag=1)
    print('DONE', time() - now)
    if dump & 4:
        pprint(D)
    if dump & 1:
        print('============== reformatting')
        p = pprettyprint(D)
        print(p)