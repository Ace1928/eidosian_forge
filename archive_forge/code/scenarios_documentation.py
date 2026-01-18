from . import iter_suite_tests, multiply_scenarios, multiply_tests
Multiply the tests in the given suite by their declared scenarios.

    Each test must have a 'scenarios' attribute which is a list of
    (name, params) pairs.

    :param some_tests: TestSuite or Test.
    :param into_suite: A TestSuite into which the resulting tests will be
        inserted.
    