from twisted.trial import runner, unittest

Mock test module that contains a C{testSuite} method. L{runner.TestLoader}
should load the tests from the C{testSuite}, not from the C{Foo} C{TestCase}.

See L{twisted.trial.test.test_loader.LoaderTest.test_loadModuleWith_testSuite}.
