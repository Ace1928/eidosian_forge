import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
class FakeTraceWithMetaclassNoName(FakeTracedCls, metaclass=profiler.TracedMeta):
    pass