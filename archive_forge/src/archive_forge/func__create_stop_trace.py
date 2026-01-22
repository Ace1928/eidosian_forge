import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
def _create_stop_trace(self):
    return self._create_trace('wsgi-stop', '2016-10-04t11:50:30.123456')