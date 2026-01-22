import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
class TestConfigDomains(unittest.TestCase):

    def test_Bool(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(True, Bool))
        self.assertEqual(c.a, True)
        c.a = False
        self.assertEqual(c.a, False)
        c.a = 1
        self.assertEqual(c.a, True)
        c.a = 'n'
        self.assertEqual(c.a, False)
        c.a = 'T'
        self.assertEqual(c.a, True)
        c.a = 'no'
        self.assertEqual(c.a, False)
        c.a = '1'
        self.assertEqual(c.a, True)
        c.a = 0.0
        self.assertEqual(c.a, False)
        c.a = True
        self.assertEqual(c.a, True)
        c.a = 0
        self.assertEqual(c.a, False)
        c.a = 'y'
        self.assertEqual(c.a, True)
        c.a = 'F'
        self.assertEqual(c.a, False)
        c.a = 'yes'
        self.assertEqual(c.a, True)
        c.a = '0'
        self.assertEqual(c.a, False)
        c.a = 1.0
        self.assertEqual(c.a, True)
        with self.assertRaises(ValueError):
            c.a = 2
        self.assertEqual(c.a, True)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, True)
        with self.assertRaises(ValueError):
            c.a = 0.5
        self.assertEqual(c.a, True)

    def test_Integer(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, Integer))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = '10'
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 10)
        with self.assertRaises(ValueError):
            c.a = '1.1'
        self.assertEqual(c.a, 10)

    def test_PositiveInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, PositiveInt))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 6)

    def test_NegativeInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NegativeInt))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -6)

    def test_NonPositiveInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NonPositiveInt))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeInt(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, NonNegativeInt))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_PositiveFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, PositiveFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 2.6)

    def test_NegativeFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NegativeFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -2.6)

    def test_NonPositiveFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(-5, NonPositiveFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.0
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeFloat(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(5, NonNegativeFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.0
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_In(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(None, In([1, 3, 5])))
        self.assertEqual(c.get('a').domain_name(), 'In[1, 3, 5]')
        self.assertEqual(c.a, None)
        c.a = 3
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = 2
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = {}
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = '1'
        self.assertEqual(c.a, 3)
        c.declare('b', ConfigValue(None, In([1, 3, 5], int)))
        self.assertEqual(c.b, None)
        c.b = 3
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = 2
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = {}
        self.assertEqual(c.b, 3)
        c.b = '1'
        self.assertEqual(c.b, 1)

        class Container(object):

            def __init__(self, vals):
                self._vals = vals

            def __str__(self):
                return f'Container{self._vals}'

            def __contains__(self, val):
                return val in self._vals
        c.declare('c', ConfigValue(None, In(Container([1, 3, 5]))))
        self.assertEqual(c.get('c').domain_name(), 'In(Container[1, 3, 5])')
        self.assertEqual(c.c, None)
        c.c = 3
        self.assertEqual(c.c, 3)
        with self.assertRaises(ValueError):
            c.c = 2
        self.assertEqual(c.c, 3)

    def test_In_enum(self):

        class TestEnum(enum.Enum):
            ITEM_ONE = 1
            ITEM_TWO = 'two'
        cfg = ConfigDict()
        cfg.declare('enum', ConfigValue(default=TestEnum.ITEM_TWO, domain=In(TestEnum)))
        self.assertEqual(cfg.get('enum').domain_name(), 'InEnum[TestEnum]')
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 'ITEM_ONE'
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = TestEnum.ITEM_TWO
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 1
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = 'two'
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
            cfg.enum = 3
        with self.assertRaisesRegex(ValueError, '.*invalid value'):
            cfg.enum = 'ITEM_THREE'

    def test_IsInstance(self):
        c = ConfigDict()
        c.declare('val', ConfigValue(None, IsInstance(int)))
        c.val = 1
        self.assertEqual(c.val, 1)
        exc_str = "Expected an instance of 'int', but received value 2.4 of type 'float'"
        with self.assertRaisesRegex(ValueError, exc_str):
            c.val = 2.4

        class TestClass:

            def __repr__(self):
                return f'{TestClass.__name__}()'
        c.declare('val2', ConfigValue(None, IsInstance(TestClass)))
        testinst = TestClass()
        c.val2 = testinst
        self.assertEqual(c.val2, testinst)
        exc_str = "Expected an instance of 'TestClass', but received value 2.4 of type 'float'"
        with self.assertRaisesRegex(ValueError, exc_str):
            c.val2 = 2.4
        c.declare('val3', ConfigValue(None, IsInstance(int, TestClass, document_full_base_names=True)))
        self.assertRegex(c.get('val3').domain_name(), 'IsInstance\\(int, .*\\.TestClass\\)')
        c.val3 = 2
        self.assertEqual(c.val3, 2)
        exc_str = "Expected an instance of one of these types: 'int', '.*\\.TestClass', but received value 2.4 of type 'float'"
        with self.assertRaisesRegex(ValueError, exc_str):
            c.val3 = 2.4
        c.declare('val4', ConfigValue(None, IsInstance(int, TestClass, document_full_base_names=False)))
        self.assertEqual(c.get('val4').domain_name(), 'IsInstance(int, TestClass)')
        c.val4 = 2
        self.assertEqual(c.val4, 2)
        exc_str = "Expected an instance of one of these types: 'int', 'TestClass', but received value 2.4 of type 'float'"
        with self.assertRaisesRegex(ValueError, exc_str):
            c.val4 = 2.4

    def test_Path(self):

        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/', os.path.sep)

        class ExamplePathLike:

            def __init__(self, path_str_or_bytes):
                self.path = path_str_or_bytes

            def __fspath__(self):
                return self.path

            def __str__(self):
                path_str = str(self.path)
                return f'{type(self).__name__}({path_str})'
        self.assertEqual(Path().domain_name(), 'Path')
        cwd = os.getcwd() + os.path.sep
        c = ConfigDict()
        c.declare('a', ConfigValue(None, Path()))
        self.assertEqual(c.a, None)
        c.a = '/a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm('/a/b/c'))
        c.a = b'/a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm('/a/b/c'))
        c.a = ExamplePathLike('/a/b/c')
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm('/a/b/c'))
        c.a = 'a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = b'a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = ExamplePathLike('a/b/c')
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = b'${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd + 'a/b/c'))
        c.a = None
        self.assertIs(c.a, None)
        c.declare('b', ConfigValue(None, Path('rel/path')))
        self.assertEqual(c.b, None)
        c.b = '/a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm('/a/b/c'))
        c.b = b'/a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm('/a/b/c'))
        c.b = ExamplePathLike('/a/b/c')
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm('/a/b/c'))
        c.b = 'a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
        c.b = b'a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
        c.b = ExamplePathLike('a/b/c')
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'rel/path/a/b/c'))
        c.b = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'a/b/c'))
        c.b = b'${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'a/b/c'))
        c.b = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd + 'a/b/c'))
        c.b = None
        self.assertIs(c.b, None)
        c.declare('c', ConfigValue(None, Path('/my/dir')))
        self.assertEqual(c.c, None)
        c.c = '/a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/a/b/c'))
        c.c = b'/a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/a/b/c'))
        c.c = ExamplePathLike('/a/b/c')
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/a/b/c'))
        c.c = 'a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/my/dir/a/b/c'))
        c.c = b'a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/my/dir/a/b/c'))
        c.c = ExamplePathLike('a/b/c')
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/my/dir/a/b/c'))
        c.c = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm(cwd + 'a/b/c'))
        c.c = b'${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm(cwd + 'a/b/c'))
        c.c = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm(cwd + 'a/b/c'))
        c.c = None
        self.assertIs(c.c, None)
        c.declare('d_base', ConfigValue('${CWD}', str))
        c.declare('d', ConfigValue(None, Path(c.get('d_base'))))
        self.assertEqual(c.d, None)
        c.d = '/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = b'/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = ExamplePathLike('/a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = 'a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = b'a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = ExamplePathLike('a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = b'${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d_base = '/my/dir'
        c.d = '/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = 'a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/my/dir/a/b/c'))
        c.d = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d_base = 'rel/path'
        c.d = '/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = b'/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = ExamplePathLike('/a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = 'a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
        c.d = b'a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
        c.d = ExamplePathLike('a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'rel/path/a/b/c'))
        c.d = '${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = b'${CWD}/a/b/c'
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        c.d = ExamplePathLike('${CWD}/a/b/c')
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd + 'a/b/c'))
        try:
            Path.SuppressPathExpansion = True
            c.d = '/a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '/a/b/c')
            c.d = b'/a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '/a/b/c')
            c.d = ExamplePathLike('/a/b/c')
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '/a/b/c')
            c.d = 'a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, 'a/b/c')
            c.d = b'a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, 'a/b/c')
            c.d = ExamplePathLike('a/b/c')
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, 'a/b/c')
            c.d = '${CWD}/a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '${CWD}/a/b/c')
            c.d = b'${CWD}/a/b/c'
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '${CWD}/a/b/c')
            c.d = ExamplePathLike('${CWD}/a/b/c')
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '${CWD}/a/b/c')
        finally:
            Path.SuppressPathExpansion = False

    def test_PathList(self):

        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/', os.path.sep)
        cwd = os.getcwd() + os.path.sep
        c = ConfigDict()
        self.assertEqual(PathList().domain_name(), 'PathList')
        c.declare('a', ConfigValue(None, PathList()))
        self.assertEqual(c.a, None)
        c.a = '/a/b/c'
        self.assertEqual(len(c.a), 1)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm('/a/b/c'))
        c.a = None
        self.assertIsNone(c.a)
        c.a = ['a/b/c', '/a/b/c', '${CWD}/a/b/c']
        self.assertEqual(len(c.a), 3)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm(cwd + 'a/b/c'))
        self.assertTrue(os.path.sep in c.a[1])
        self.assertEqual(c.a[1], norm('/a/b/c'))
        self.assertTrue(os.path.sep in c.a[2])
        self.assertEqual(c.a[2], norm(cwd + 'a/b/c'))
        c.a = ()
        self.assertEqual(len(c.a), 0)
        self.assertIs(type(c.a), list)
        exc_str = '.*expected str, bytes or os.PathLike.*int'
        with self.assertRaisesRegex(ValueError, exc_str):
            c.a = 2
        with self.assertRaisesRegex(ValueError, exc_str):
            c.a = ['/a/b/c', 2]

    def test_ListOf(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(domain=ListOf(int), default=None))
        self.assertEqual(c.get('a').domain_name(), 'ListOf[int]')
        self.assertEqual(c.a, None)
        c.a = 5
        self.assertEqual(c.a, [5])
        c.a = (5, 6.6)
        self.assertEqual(c.a, [5, 6])
        c.a = '7,8'
        self.assertEqual(c.a, [7, 8])
        ref = "(?m)Failed casting a\\s+to ListOf\\(int\\)\\s+Error: invalid literal for int\\(\\) with base 10: 'a'"
        with self.assertRaisesRegex(ValueError, ref):
            c.a = 'a'
        c.declare('b', ConfigValue(domain=ListOf(str), default=None))
        self.assertEqual(c.get('b').domain_name(), 'ListOf[str]')
        self.assertEqual(c.b, None)
        c.b = "'Hello, World'"
        self.assertEqual(c.b, ['Hello, World'])
        c.b = 'Hello, World'
        self.assertEqual(c.b, ['Hello', 'World'])
        c.b = ('A', 6)
        self.assertEqual(c.b, ['A', '6'])
        with self.assertRaises(ValueError):
            c.b = "'Hello, World"
        c.declare('b1', ConfigValue(domain=ListOf(str, string_lexer=None), default=None))
        self.assertEqual(c.get('b1').domain_name(), 'ListOf[str]')
        self.assertEqual(c.b1, None)
        c.b1 = "'Hello, World'"
        self.assertEqual(c.b1, ["'Hello, World'"])
        c.b1 = 'Hello, World'
        self.assertEqual(c.b1, ['Hello, World'])
        c.b1 = ('A', 6)
        self.assertEqual(c.b1, ['A', '6'])
        c.b1 = "'Hello, World"
        self.assertEqual(c.b1, ["'Hello, World"])
        c.declare('c', ConfigValue(domain=ListOf(int, PositiveInt)))
        self.assertEqual(c.get('c').domain_name(), 'ListOf[PositiveInt]')
        self.assertEqual(c.c, None)
        c.c = 6
        self.assertEqual(c.c, [6])
        ref = '(?m)Failed casting %s\\s+to ListOf\\(PositiveInt\\)\\s+Error: Expected positive int, but received %s'
        with self.assertRaisesRegex(ValueError, ref % (6.5, 6.5)):
            c.c = 6.5
        with self.assertRaisesRegex(ValueError, ref % ('\\[0\\]', '0')):
            c.c = [0]
        c.c = [3, 6, 9]
        self.assertEqual(c.c, [3, 6, 9])

    def test_Module(self):
        c = ConfigDict()
        c.declare('a', ConfigValue(domain=Module(), default=None))
        self.assertEqual(c.a, None)
        c.a = 'os.path'
        import os.path
        self.assertIs(c.a, os.path)
        import os
        c.a = os
        self.assertIs(c.a, os)
        this_file = __file__
        this_folder = os.path.dirname(__file__)
        to_import = os.path.join(this_folder, 'test_config.py')
        c.a = to_import
        self.assertEqual(c.a.__file__, to_import)

    def test_ConfigEnum(self):
        out = StringIO()
        with LoggingIntercept(out):

            class TestEnum(ConfigEnum):
                ITEM_ONE = 1
                ITEM_TWO = 2
        self.assertIn('The ConfigEnum base class is deprecated', out.getvalue())
        self.assertEqual(TestEnum.from_enum_or_string(1), TestEnum.ITEM_ONE)
        self.assertEqual(TestEnum.from_enum_or_string(TestEnum.ITEM_TWO), TestEnum.ITEM_TWO)
        self.assertEqual(TestEnum.from_enum_or_string('ITEM_ONE'), TestEnum.ITEM_ONE)
        cfg = ConfigDict()
        cfg.declare('enum', ConfigValue(default=2, domain=TestEnum.from_enum_or_string))
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 'ITEM_ONE'
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        cfg.enum = TestEnum.ITEM_TWO
        self.assertEqual(cfg.enum, TestEnum.ITEM_TWO)
        cfg.enum = 1
        self.assertEqual(cfg.enum, TestEnum.ITEM_ONE)
        with self.assertRaisesRegex(ValueError, '.*3 is not a valid'):
            cfg.enum = 3
        with self.assertRaisesRegex(ValueError, '.*invalid value'):
            cfg.enum = 'ITEM_THREE'

    def test_DynamicImplicitDomain(self):

        def _rule(key, val):
            ans = ConfigDict()
            if 'i' in key:
                ans.declare('option_i', ConfigValue(domain=int, default=1))
            if 'f' in key:
                ans.declare('option_f', ConfigValue(domain=float, default=2))
            if 's' in key:
                ans.declare('option_s', ConfigValue(domain=str, default=3))
            if 'l' in key:
                raise ValueError('invalid key: %s' % key)
            return ans(val)
        cfg = ConfigDict(implicit=True, implicit_domain=DynamicImplicitDomain(_rule))
        self.assertEqual(len(cfg), 0)
        test = cfg({'hi': {'option_i': 10}, 'fast': {'option_f': 20}})
        self.assertEqual(len(test), 2)
        self.assertEqual(test.hi.value(), {'option_i': 10})
        self.assertEqual(test.fast.value(), {'option_f': 20, 'option_s': '3'})
        test2 = cfg(test)
        self.assertIsNot(test.hi, test2.hi)
        self.assertIsNot(test.fast, test2.fast)
        self.assertEqual(test.value(), test2.value())
        self.assertEqual(len(test2), 2)
        fit = test2.get('fit', {})
        self.assertEqual(len(test2), 2)
        self.assertEqual(fit.value(), {'option_f': 2, 'option_i': 1})
        with self.assertRaisesRegex(ValueError, 'invalid key: fail'):
            test = cfg({'hi': {'option_i': 10}, 'fast': {'option_f': 20}, 'fail': {'option_f': 20}})