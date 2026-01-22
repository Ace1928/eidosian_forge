from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
class TestRunTest(TestCase):
    run_tests_with = FullStackRunTest

    def make_case(self):

        class Case(TestCase):

            def test(self):
                pass
        return Case('test')

    def test___init___short(self):
        run = RunTest('bar')
        self.assertEqual('bar', run.case)
        self.assertEqual([], run.handlers)

    def test__init____handlers(self):
        handlers = [('quux', 'baz')]
        run = RunTest('bar', handlers)
        self.assertEqual(handlers, run.handlers)

    def test__init____handlers_last_resort(self):
        handlers = [('quux', 'baz')]
        last_resort = 'foo'
        run = RunTest('bar', handlers, last_resort)
        self.assertEqual(last_resort, run.last_resort)

    def test_run_with_result(self):
        log = []

        class Case(TestCase):

            def _run_test_method(self, result):
                log.append(result)
        case = Case('_run_test_method')
        run = RunTest(case, lambda x: log.append(x))
        result = TestResult()
        run.run(result)
        self.assertEqual(1, len(log))
        self.assertEqual(result, log[0].decorated)

    def test_run_no_result_manages_new_result(self):
        log = []
        run = RunTest(self.make_case(), lambda x: log.append(x) or x)
        result = run.run()
        self.assertIsInstance(result.decorated, TestResult)

    def test__run_core_called(self):
        case = self.make_case()
        log = []
        run = RunTest(case, lambda x: x)
        run._run_core = lambda: log.append('foo')
        run.run()
        self.assertEqual(['foo'], log)

    def test__run_prepared_result_does_not_mask_keyboard(self):
        tearDownRuns = []

        class Case(TestCase):

            def test(self):
                raise KeyboardInterrupt('go')

            def _run_teardown(self, result):
                tearDownRuns.append(self)
                return super()._run_teardown(result)
        case = Case('test')
        run = RunTest(case)
        run.result = ExtendedTestResult()
        self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(KeyboardInterrupt)))
        self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
        self.assertThat(tearDownRuns, HasLength(1))

    def test__run_user_calls_onException(self):
        case = self.make_case()
        log = []

        def handler(exc_info):
            log.append('got it')
            self.assertEqual(3, len(exc_info))
            self.assertIsInstance(exc_info[1], KeyError)
            self.assertIs(KeyError, exc_info[0])
        case.addOnException(handler)
        e = KeyError('Yo')

        def raises():
            raise e
        run = RunTest(case, [(KeyError, None)])
        run.result = ExtendedTestResult()
        status = run._run_user(raises)
        self.assertEqual(run.exception_caught, status)
        self.assertEqual([], run.result._events)
        self.assertEqual(['got it'], log)

    def test__run_user_can_catch_Exception(self):
        case = self.make_case()
        e = Exception('Yo')

        def raises():
            raise e
        log = []
        run = RunTest(case, [(Exception, None)])
        run.result = ExtendedTestResult()
        status = run._run_user(raises)
        self.assertEqual(run.exception_caught, status)
        self.assertEqual([], run.result._events)
        self.assertEqual([], log)

    def test__run_prepared_result_uncaught_Exception_raised(self):
        e = KeyError('Yo')

        class Case(TestCase):

            def test(self):
                raise e
        case = Case('test')
        log = []

        def log_exc(self, result, err):
            log.append((result, err))
        run = RunTest(case, [(ValueError, log_exc)])
        run.result = ExtendedTestResult()
        self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(KeyError)))
        self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
        self.assertEqual([], log)

    def test__run_prepared_result_uncaught_Exception_triggers_error(self):
        e = SystemExit(0)

        class Case(TestCase):

            def test(self):
                raise e
        case = Case('test')
        log = []

        def log_exc(self, result, err):
            log.append((result, err))
        run = RunTest(case, [], log_exc)
        run.result = ExtendedTestResult()
        self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(SystemExit)))
        self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
        self.assertEqual([(run.result, e)], log)

    def test__run_user_uncaught_Exception_from_exception_handler_raised(self):
        case = self.make_case()

        def broken_handler(exc_info):
            raise ValueError('boo')
        case.addOnException(broken_handler)
        e = KeyError('Yo')

        def raises():
            raise e
        log = []

        def log_exc(self, result, err):
            log.append((result, err))
        run = RunTest(case, [(ValueError, log_exc)])
        run.result = ExtendedTestResult()
        self.assertThat(lambda: run._run_user(raises), Raises(MatchesException(ValueError)))
        self.assertEqual([], run.result._events)
        self.assertEqual([], log)

    def test__run_user_returns_result(self):
        case = self.make_case()

        def returns():
            return 1
        run = RunTest(case)
        run.result = ExtendedTestResult()
        self.assertEqual(1, run._run_user(returns))
        self.assertEqual([], run.result._events)

    def test__run_one_decorates_result(self):
        log = []

        class Run(RunTest):

            def _run_prepared_result(self, result):
                log.append(result)
                return result
        run = Run(self.make_case(), lambda x: x)
        result = run._run_one('foo')
        self.assertEqual([result], log)
        self.assertIsInstance(log[0], ExtendedToOriginalDecorator)
        self.assertEqual('foo', result.decorated)

    def test__run_prepared_result_calls_start_and_stop_test(self):
        result = ExtendedTestResult()
        case = self.make_case()
        run = RunTest(case, lambda x: x)
        run.run(result)
        self.assertEqual([('startTest', case), ('addSuccess', case), ('stopTest', case)], result._events)

    def test__run_prepared_result_calls_stop_test_always(self):
        result = ExtendedTestResult()
        case = self.make_case()

        def inner():
            raise Exception('foo')
        run = RunTest(case, lambda x: x)
        run._run_core = inner
        self.assertThat(lambda: run.run(result), Raises(MatchesException(Exception('foo'))))
        self.assertEqual([('startTest', case), ('stopTest', case)], result._events)