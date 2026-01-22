from __future__ import absolute_import
import unittest
import sys
import os
def all_tests_suite(project_dir=None):

    def get_suite():
        suite_names = ['simplejson.tests.%s' % (os.path.splitext(f)[0],) for f in os.listdir(os.path.dirname(__file__)) if f.startswith('test_') and f.endswith('.py')]
        return additional_tests(suite=unittest.TestLoader().loadTestsFromNames(suite_names), project_dir=project_dir)
    suite = get_suite()
    import simplejson
    if simplejson._import_c_make_encoder() is None:
        suite.addTest(TestMissingSpeedups())
    else:
        suite = unittest.TestSuite([suite, NoExtensionTestSuite([get_suite()])])
    return suite