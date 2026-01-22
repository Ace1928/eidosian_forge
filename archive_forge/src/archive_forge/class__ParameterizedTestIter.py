from the command line::
from collections import abc
import functools
import inspect
import itertools
import re
import types
import unittest
import warnings
from absl.testing import absltest
class _ParameterizedTestIter(object):
    """Callable and iterable class for producing new test cases."""

    def __init__(self, test_method, testcases, naming_type, original_name=None):
        """Returns concrete test functions for a test and a list of parameters.

    The naming_type is used to determine the name of the concrete
    functions as reported by the unittest framework. If naming_type is
    _FIRST_ARG, the testcases must be tuples, and the first element must
    have a string representation that is a valid Python identifier.

    Args:
      test_method: The decorated test method.
      testcases: (list of tuple/dict) A list of parameter tuples/dicts for
          individual test invocations.
      naming_type: The test naming type, either _NAMED or _ARGUMENT_REPR.
      original_name: The original test method name. When decorated on a test
          method, None is passed to __init__ and test_method.__name__ is used.
          Note test_method.__name__ might be different than the original defined
          test method because of the use of other decorators. A more accurate
          value is set by TestGeneratorMetaclass.__new__ later.
    """
        self._test_method = test_method
        self.testcases = testcases
        self._naming_type = naming_type
        if original_name is None:
            original_name = test_method.__name__
        self._original_name = original_name
        self.__name__ = _ParameterizedTestIter.__name__

    def __call__(self, *args, **kwargs):
        raise RuntimeError('You appear to be running a parameterized test case without having inherited from parameterized.TestCase. This is bad because none of your test cases are actually being run. You may also be using another decorator before the parameterized one, in which case you should reverse the order.')

    def __iter__(self):
        test_method = self._test_method
        naming_type = self._naming_type

        def make_bound_param_test(testcase_params):

            @functools.wraps(test_method)
            def bound_param_test(self):
                if isinstance(testcase_params, abc.Mapping):
                    return test_method(self, **testcase_params)
                elif _non_string_or_bytes_iterable(testcase_params):
                    return test_method(self, *testcase_params)
                else:
                    return test_method(self, testcase_params)
            if naming_type is _NAMED:
                bound_param_test.__x_use_name__ = True
                testcase_name = None
                if isinstance(testcase_params, abc.Mapping):
                    if _NAMED_DICT_KEY not in testcase_params:
                        raise RuntimeError('Dict for named tests must contain key "%s"' % _NAMED_DICT_KEY)
                    testcase_name = testcase_params[_NAMED_DICT_KEY]
                    testcase_params = {k: v for k, v in testcase_params.items() if k != _NAMED_DICT_KEY}
                elif _non_string_or_bytes_iterable(testcase_params):
                    if not isinstance(testcase_params[0], str):
                        raise RuntimeError('The first element of named test parameters is the test name suffix and must be a string')
                    testcase_name = testcase_params[0]
                    testcase_params = testcase_params[1:]
                else:
                    raise RuntimeError('Named tests must be passed a dict or non-string iterable.')
                test_method_name = self._original_name
                if test_method_name.startswith('test_') and testcase_name and (not testcase_name.startswith('_')):
                    test_method_name += '_'
                bound_param_test.__name__ = test_method_name + str(testcase_name)
            elif naming_type is _ARGUMENT_REPR:
                if isinstance(testcase_params, types.GeneratorType):
                    testcase_params = tuple(testcase_params)
                params_repr = '(%s)' % (_format_parameter_list(testcase_params),)
                bound_param_test.__x_params_repr__ = params_repr
            else:
                raise RuntimeError('%s is not a valid naming type.' % (naming_type,))
            bound_param_test.__doc__ = '%s(%s)' % (bound_param_test.__name__, _format_parameter_list(testcase_params))
            if test_method.__doc__:
                bound_param_test.__doc__ += '\n%s' % (test_method.__doc__,)
            if inspect.iscoroutinefunction(test_method):
                return _async_wrapped(bound_param_test)
            return bound_param_test
        return (make_bound_param_test(c) for c in self.testcases)