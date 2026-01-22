from __future__ import absolute_import
import unittest
import sys
import os
def additional_tests(suite=None, project_dir=None):
    import simplejson
    import simplejson.encoder
    import simplejson.decoder
    if suite is None:
        suite = unittest.TestSuite()
    try:
        import doctest
    except ImportError:
        if sys.version_info < (2, 7):
            return suite
        raise
    for mod in (simplejson, simplejson.encoder, simplejson.decoder):
        suite.addTest(doctest.DocTestSuite(mod))
    if project_dir is not None:
        suite.addTest(doctest.DocFileSuite(os.path.join(project_dir, 'index.rst'), module_relative=False))
    return suite