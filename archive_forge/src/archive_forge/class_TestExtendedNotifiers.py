import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class TestExtendedNotifiers(unittest.TestCase):

    def setUp(self):
        self.exceptions = []
        trait_notifiers.push_exception_handler(self._handle_exception)

    def tearDown(self):
        trait_notifiers.pop_exception_handler()

    def _handle_exception(self, obj, name, old, new):
        self.exceptions.append((obj, name, old, new))

    def test_extended_notifiers_methods(self):
        obj = ExtendedNotifiers(ok=2)
        obj.ok = 3
        self.assertEqual(len(obj.rebind_calls_0), 2)
        expected_1 = [2, 3]
        self.assertEqual(expected_1, obj.rebind_calls_1)
        expected_2 = [('ok', 2), ('ok', 3)]
        self.assertEqual(expected_2, obj.rebind_calls_2)
        expected_3 = [(obj, 'ok', 2), (obj, 'ok', 3)]
        self.assertEqual(expected_3, obj.rebind_calls_3)
        expected_4 = [(obj, 'ok', 0, 2), (obj, 'ok', 2, 3)]
        self.assertEqual(expected_4, obj.rebind_calls_4)

    def test_extended_notifiers_methods_failing(self):
        obj = ExtendedNotifiers()
        obj.fail = 1
        self.assertCountEqual([0, 1, 2, 3, 4], obj.exceptions_from)
        self.assertEqual([(obj, 'fail', 0, 1)] * 5, self.exceptions)

    def test_extended_notifiers_functions(self):
        calls_0.clear()
        calls_1.clear()
        calls_2.clear()
        calls_3.clear()
        calls_4.clear()
        obj = ExtendedNotifiers()
        obj._on_trait_change(function_listener_0, 'ok', dispatch='extended')
        obj._on_trait_change(function_listener_1, 'ok', dispatch='extended')
        obj._on_trait_change(function_listener_2, 'ok', dispatch='extended')
        obj._on_trait_change(function_listener_3, 'ok', dispatch='extended')
        obj._on_trait_change(function_listener_4, 'ok', dispatch='extended')
        obj.ok = 2
        obj.ok = 3
        expected_0 = [True, True]
        self.assertEqual(expected_0, calls_0)
        expected_1 = [2, 3]
        self.assertEqual(expected_1, calls_1)
        expected_2 = [('ok', 2), ('ok', 3)]
        self.assertEqual(expected_2, calls_2)
        expected_3 = [(obj, 'ok', 2), (obj, 'ok', 3)]
        self.assertEqual(expected_3, calls_3)
        expected_4 = [(obj, 'ok', 0, 2), (obj, 'ok', 2, 3)]
        self.assertEqual(expected_4, calls_4)

    def test_extended_notifiers_functions_failing(self):
        obj = ExtendedNotifiers()
        exceptions_from.clear()
        obj._on_trait_change(failing_function_listener_0, 'fail', dispatch='extended')
        obj._on_trait_change(failing_function_listener_1, 'fail', dispatch='extended')
        obj._on_trait_change(failing_function_listener_2, 'fail', dispatch='extended')
        obj._on_trait_change(failing_function_listener_3, 'fail', dispatch='extended')
        obj._on_trait_change(failing_function_listener_4, 'fail', dispatch='extended')
        obj.fail = 1
        self.assertCountEqual([0, 1, 2, 3, 4], obj.exceptions_from)
        self.assertEqual([(obj, 'fail', 0, 1)] * 10, self.exceptions)