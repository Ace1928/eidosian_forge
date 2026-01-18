import unittest
from unittest import mock
def test_seal_with_autospec(self):

    class Foo:
        foo = 0

        def bar1(self):
            return 1

        def bar2(self):
            return 2

        class Baz:
            baz = 3

            def ban(self):
                return 4
    for spec_set in (True, False):
        with self.subTest(spec_set=spec_set):
            foo = mock.create_autospec(Foo, spec_set=spec_set)
            foo.bar1.return_value = 'a'
            foo.Baz.ban.return_value = 'b'
            mock.seal(foo)
            self.assertIsInstance(foo.foo, mock.NonCallableMagicMock)
            self.assertIsInstance(foo.bar1, mock.MagicMock)
            self.assertIsInstance(foo.bar2, mock.MagicMock)
            self.assertIsInstance(foo.Baz, mock.MagicMock)
            self.assertIsInstance(foo.Baz.baz, mock.NonCallableMagicMock)
            self.assertIsInstance(foo.Baz.ban, mock.MagicMock)
            self.assertEqual(foo.bar1(), 'a')
            foo.bar1.return_value = 'new_a'
            self.assertEqual(foo.bar1(), 'new_a')
            self.assertEqual(foo.Baz.ban(), 'b')
            foo.Baz.ban.return_value = 'new_b'
            self.assertEqual(foo.Baz.ban(), 'new_b')
            with self.assertRaises(TypeError):
                foo.foo()
            with self.assertRaises(AttributeError):
                foo.bar = 1
            with self.assertRaises(AttributeError):
                foo.bar2()
            foo.bar2.return_value = 'bar2'
            self.assertEqual(foo.bar2(), 'bar2')
            with self.assertRaises(AttributeError):
                foo.missing_attr
            with self.assertRaises(AttributeError):
                foo.missing_attr = 1
            with self.assertRaises(AttributeError):
                foo.missing_method()
            with self.assertRaises(TypeError):
                foo.Baz.baz()
            with self.assertRaises(AttributeError):
                foo.Baz.missing_attr
            with self.assertRaises(AttributeError):
                foo.Baz.missing_attr = 1
            with self.assertRaises(AttributeError):
                foo.Baz.missing_method()