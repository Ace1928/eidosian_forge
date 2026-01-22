import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class TraceWithMetaclassTestCase(test.TestCase):

    def test_no_name_exception(self):

        def define_class_with_no_name():

            class FakeTraceWithMetaclassNoName(FakeTracedCls, metaclass=profiler.TracedMeta):
                pass
        self.assertRaises(TypeError, define_class_with_no_name, 1)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_args(self, mock_start, mock_stop):
        fake_cls = FakeTraceWithMetaclassBase()
        self.assertEqual(30, fake_cls.method1(5, 15))
        expected_info = {'a': 10, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassBase.method1', 'args': str((fake_cls, 5, 15)), 'kwargs': str({})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_kwargs(self, mock_start, mock_stop):
        fake_cls = FakeTraceWithMetaclassBase()
        self.assertEqual(50, fake_cls.method3(g=5, h=10))
        expected_info = {'a': 10, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassBase.method3', 'args': str((fake_cls,)), 'kwargs': str({'g': 5, 'h': 10})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_without_private(self, mock_start, mock_stop):
        fake_cls = FakeTraceWithMetaclassHideArgs()
        self.assertEqual(10, fake_cls._method(10))
        self.assertFalse(mock_start.called)
        self.assertFalse(mock_stop.called)

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_without_args(self, mock_start, mock_stop):
        fake_cls = FakeTraceWithMetaclassHideArgs()
        self.assertEqual(20, fake_cls.method5(5, 15))
        expected_info = {'b': 20, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassHideArgs.method5'}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('a', expected_info))
        mock_stop.assert_called_once_with()

    @mock.patch('osprofiler.profiler.stop')
    @mock.patch('osprofiler.profiler.start')
    def test_private_methods(self, mock_start, mock_stop):
        fake_cls = FakeTraceWithMetaclassPrivate()
        self.assertEqual(10, fake_cls._new_private_method(5))
        expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassPrivate._new_private_method', 'args': str((fake_cls, 5)), 'kwargs': str({})}}
        self.assertEqual(1, len(mock_start.call_args_list))
        self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
        mock_stop.assert_called_once_with()