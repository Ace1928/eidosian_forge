import sys
from greenlet import greenlet
from . import TestCase
def test_throw_goes_to_original_parent(self):
    main = greenlet.getcurrent()

    def f1():
        try:
            main.switch('f1 ready to catch')
        except IndexError:
            return 'caught'
        return 'normal exit'

    def f2():
        main.switch('from f2')
    g1 = greenlet(f1)
    g2 = greenlet(f2, parent=g1)
    with self.assertRaises(IndexError):
        g2.throw(IndexError)
    self.assertTrue(g2.dead)
    self.assertTrue(g1.dead)
    g1 = greenlet(f1)
    g2 = greenlet(f2, parent=g1)
    res = g1.switch()
    self.assertEqual(res, 'f1 ready to catch')
    res = g2.throw(IndexError)
    self.assertEqual(res, 'caught')
    self.assertTrue(g2.dead)
    self.assertTrue(g1.dead)
    g1 = greenlet(f1)
    g2 = greenlet(f2, parent=g1)
    res = g1.switch()
    self.assertEqual(res, 'f1 ready to catch')
    res = g2.switch()
    self.assertEqual(res, 'from f2')
    res = g2.throw(IndexError)
    self.assertEqual(res, 'caught')
    self.assertTrue(g2.dead)
    self.assertTrue(g1.dead)