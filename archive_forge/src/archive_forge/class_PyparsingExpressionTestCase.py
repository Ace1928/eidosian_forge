from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class PyparsingExpressionTestCase(unittest.TestCase):
    """
    Base pyparsing testing class to parse various pyparsing expressions against
    given text strings. Subclasses must define a class attribute 'tests' which
    is a list of PpTestSpec instances.
    """
    if not hasattr(unittest.TestCase, 'subTest'):
        from contextlib import contextmanager

        @contextmanager
        def subTest(self, **params):
            print('subTest:', params)
            yield
    tests = []

    def runTest(self):
        if self.__class__ is PyparsingExpressionTestCase:
            return
        for test_spec in self.tests:
            with self.subTest(test_spec=test_spec):
                test_spec.expr.streamline()
                print('\n{0} - {1}({2})'.format(test_spec.desc, type(test_spec.expr).__name__, test_spec.expr))
                parsefn = getattr(test_spec.expr, test_spec.parse_fn)
                if test_spec.expected_fail_locn is None:
                    result = parsefn(test_spec.text)
                    if test_spec.parse_fn == 'parseString':
                        print(result.dump())
                        if test_spec.expected_list is not None:
                            self.assertEqual(result.asList(), test_spec.expected_list)
                        if test_spec.expected_dict is not None:
                            self.assertEqual(result.asDict(), test_spec.expected_dict)
                    elif test_spec.parse_fn == 'transformString':
                        print(result)
                        if test_spec.expected_list is not None:
                            self.assertEqual([result], test_spec.expected_list)
                    elif test_spec.parse_fn == 'searchString':
                        print(result)
                        if test_spec.expected_list is not None:
                            self.assertEqual([result], test_spec.expected_list)
                else:
                    try:
                        parsefn(test_spec.text)
                    except Exception as exc:
                        if not hasattr(exc, '__traceback__'):
                            from sys import exc_info
                            etype, value, traceback = exc_info()
                            exc.__traceback__ = traceback
                        print(pp.ParseException.explain(exc))
                        self.assertEqual(exc.loc, test_spec.expected_fail_locn)
                    else:
                        self.assertTrue(False, 'failed to raise expected exception')