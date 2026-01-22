import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class MyLogFormatter(test_log.LogCatcher):

    def __new__(klass, *args, **kwargs):
        self.log_catcher = test_log.LogCatcher(*args, **kwargs)
        return self.log_catcher