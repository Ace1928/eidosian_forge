from mpmath import *
from mpmath import fp

Easy-to-use test-generating code:

cases = '''
exp 2.25
log 2.25
'''

from mpmath import *
mp.dps = 20
for test in cases.splitlines():
    if not test:
        continue
    words = test.split()
    fname = words[0]
    args = words[1:]
    argstr = ", ".join(args)
    testline = "%s(%s)" % (fname, argstr)
    ans = str(eval(testline))
    print "    assert ae(fp.%s, %s)" % (testline, ans)

