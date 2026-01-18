import sys
from greenlet import greenlet
from . import TestCase
def test_val(self):

    def f():
        try:
            switch('ok')
        except RuntimeError:
            val = sys.exc_info()[1]
            if str(val) == 'ciao':
                switch('ok')
                return
        switch('fail')
    g = greenlet(f)
    res = g.switch()
    self.assertEqual(res, 'ok')
    res = g.throw(RuntimeError('ciao'))
    self.assertEqual(res, 'ok')
    g = greenlet(f)
    res = g.switch()
    self.assertEqual(res, 'ok')
    res = g.throw(RuntimeError, 'ciao')
    self.assertEqual(res, 'ok')