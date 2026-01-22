from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
class ConcurrentStreamTestSuite:
    """A TestSuite whose run() parallelises."""

    def __init__(self, make_tests):
        """Create a ConcurrentTestSuite to execute tests returned by make_tests.

        :param make_tests: A helper function that should return some number
            of concurrently executable test suite / test case objects.
            make_tests must take no parameters and return an iterable of
            tuples. Each tuple must be of the form (case, route_code), where
            case is a TestCase-like object with a run(result) method, and
            route_code is either None or a unicode string.
        """
        super().__init__()
        self.make_tests = make_tests

    def run(self, result):
        """Run the tests concurrently.

        This calls out to the provided make_tests helper to determine the
        concurrency to use and to assign routing codes to each worker.

        ConcurrentTestSuite provides no special mechanism to stop the tests
        returned by make_tests, it is up to the made tests to honour the
        shouldStop attribute on the result object they are run with, which will
        be set if the test run is to be aborted.

        The tests are run with an ExtendedToStreamDecorator wrapped around a
        StreamToQueue instance. ConcurrentStreamTestSuite dequeues events from
        the queue and forwards them to result. Tests can therefore be either
        original unittest tests (or compatible tests), or new tests that emit
        StreamResult events directly.

        :param result: A StreamResult instance. The caller is responsible for
            calling startTestRun on this instance prior to invoking suite.run,
            and stopTestRun subsequent to the run method returning.
        """
        tests = self.make_tests()
        try:
            threads = {}
            queue = Queue()
            for test, route_code in tests:
                to_queue = testtools.StreamToQueue(queue, route_code)
                process_result = testtools.ExtendedToStreamDecorator(testtools.TimestampingStreamResult(to_queue))
                runner_thread = threading.Thread(target=self._run_test, args=(test, process_result, route_code))
                threads[to_queue] = (runner_thread, process_result)
                runner_thread.start()
            while threads:
                event_dict = queue.get()
                event = event_dict.pop('event')
                if event == 'status':
                    result.status(**event_dict)
                elif event == 'stopTestRun':
                    thread = threads.pop(event_dict['result'])[0]
                    thread.join()
                elif event == 'startTestRun':
                    pass
                else:
                    raise ValueError(f'unknown event type {event!r}')
        except:
            for thread, process_result in threads.values():
                process_result.stop()
            raise

    def _run_test(self, test, process_result, route_code):
        process_result.startTestRun()
        try:
            try:
                test.run(process_result)
            except Exception:
                case = testtools.ErrorHolder(f"broken-runner-'{route_code}'", error=sys.exc_info())
                case.run(process_result)
        finally:
            process_result.stopTestRun()