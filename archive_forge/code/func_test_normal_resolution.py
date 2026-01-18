from collections import defaultdict
import unittest
from lazr.uri import (
def test_normal_resolution(self):
    base = URI('http://a/b/c/d;p?q')

    def resolve(relative):
        return str(base.resolve(relative))
    self.assertEqual(resolve('g:h'), 'g:h')
    self.assertEqual(resolve('g'), 'http://a/b/c/g')
    self.assertEqual(resolve('./g'), 'http://a/b/c/g')
    self.assertEqual(resolve('g/'), 'http://a/b/c/g/')
    self.assertEqual(resolve('/g'), 'http://a/g')
    self.assertEqual(resolve('//g'), 'http://g/')
    self.assertEqual(resolve('?y'), 'http://a/b/c/d;p?y')
    self.assertEqual(resolve('g?y'), 'http://a/b/c/g?y')
    self.assertEqual(resolve('#s'), 'http://a/b/c/d;p?q#s')
    self.assertEqual(resolve('g#s'), 'http://a/b/c/g#s')
    self.assertEqual(resolve('g?y#s'), 'http://a/b/c/g?y#s')
    self.assertEqual(resolve(';x'), 'http://a/b/c/;x')
    self.assertEqual(resolve('g;x'), 'http://a/b/c/g;x')
    self.assertEqual(resolve('g;x?y#s'), 'http://a/b/c/g;x?y#s')
    self.assertEqual(resolve(''), 'http://a/b/c/d;p?q')
    self.assertEqual(resolve('.'), 'http://a/b/c/')
    self.assertEqual(resolve('./'), 'http://a/b/c/')
    self.assertEqual(resolve('..'), 'http://a/b/')
    self.assertEqual(resolve('../'), 'http://a/b/')
    self.assertEqual(resolve('../g'), 'http://a/b/g')
    self.assertEqual(resolve('../..'), 'http://a/')
    self.assertEqual(resolve('../../'), 'http://a/')
    self.assertEqual(resolve('../../g'), 'http://a/g')