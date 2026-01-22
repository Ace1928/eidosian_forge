from testtools import TestCase
from testtools.matchers import MatchesException, Raises
from testtools.monkey import MonkeyPatcher, patch
class MonkeyPatcherTest(TestCase):
    """
    Tests for 'MonkeyPatcher' monkey-patching class.
    """

    def setUp(self):
        super().setUp()
        self.test_object = TestObj()
        self.original_object = TestObj()
        self.monkey_patcher = MonkeyPatcher()

    def test_empty(self):
        self.monkey_patcher.patch()
        self.assertEqual(self.original_object.foo, self.test_object.foo)
        self.assertEqual(self.original_object.bar, self.test_object.bar)
        self.assertEqual(self.original_object.baz, self.test_object.baz)

    def test_construct_with_patches(self):
        patcher = MonkeyPatcher((self.test_object, 'foo', 'haha'), (self.test_object, 'bar', 'hehe'))
        patcher.patch()
        self.assertEqual('haha', self.test_object.foo)
        self.assertEqual('hehe', self.test_object.bar)
        self.assertEqual(self.original_object.baz, self.test_object.baz)

    def test_patch_existing(self):
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
        self.monkey_patcher.patch()
        self.assertEqual(self.test_object.foo, 'haha')

    def test_patch_non_existing(self):
        self.monkey_patcher.add_patch(self.test_object, 'doesntexist', 'value')
        self.monkey_patcher.patch()
        self.assertEqual(self.test_object.doesntexist, 'value')

    def test_restore_non_existing(self):
        self.monkey_patcher.add_patch(self.test_object, 'doesntexist', 'value')
        self.monkey_patcher.patch()
        self.monkey_patcher.restore()
        marker = object()
        self.assertIs(marker, getattr(self.test_object, 'doesntexist', marker))

    def test_patch_already_patched(self):
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'blah')
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'BLAH')
        self.monkey_patcher.patch()
        self.assertEqual(self.test_object.foo, 'BLAH')
        self.monkey_patcher.restore()
        self.assertEqual(self.test_object.foo, self.original_object.foo)

    def test_restore_twice_is_a_no_op(self):
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'blah')
        self.monkey_patcher.patch()
        self.monkey_patcher.restore()
        self.assertEqual(self.test_object.foo, self.original_object.foo)
        self.monkey_patcher.restore()
        self.assertEqual(self.test_object.foo, self.original_object.foo)

    def test_run_with_patches_decoration(self):
        log = []

        def f(a, b, c=None):
            log.append((a, b, c))
            return 'foo'
        result = self.monkey_patcher.run_with_patches(f, 1, 2, c=10)
        self.assertEqual('foo', result)
        self.assertEqual([(1, 2, 10)], log)

    def test_repeated_run_with_patches(self):

        def f():
            return (self.test_object.foo, self.test_object.bar, self.test_object.baz)
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
        result = self.monkey_patcher.run_with_patches(f)
        self.assertEqual(('haha', self.original_object.bar, self.original_object.baz), result)
        result = self.monkey_patcher.run_with_patches(f)
        self.assertEqual(('haha', self.original_object.bar, self.original_object.baz), result)

    def test_run_with_patches_restores(self):
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
        self.assertEqual(self.original_object.foo, self.test_object.foo)
        self.monkey_patcher.run_with_patches(lambda: None)
        self.assertEqual(self.original_object.foo, self.test_object.foo)

    def test_run_with_patches_restores_on_exception(self):

        def _():
            self.assertEqual(self.test_object.foo, 'haha')
            self.assertEqual(self.test_object.bar, 'blahblah')
            raise RuntimeError('Something went wrong!')
        self.monkey_patcher.add_patch(self.test_object, 'foo', 'haha')
        self.monkey_patcher.add_patch(self.test_object, 'bar', 'blahblah')
        self.assertThat(lambda: self.monkey_patcher.run_with_patches(_), Raises(MatchesException(RuntimeError('Something went wrong!'))))
        self.assertEqual(self.test_object.foo, self.original_object.foo)
        self.assertEqual(self.test_object.bar, self.original_object.bar)