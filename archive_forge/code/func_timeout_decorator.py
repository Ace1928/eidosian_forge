import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def timeout_decorator(fcn):

    @functools.wraps(fcn)
    def test_timer(*args, **kwargs):
        qualname = '%s.%s' % (fcn.__module__, fcn.__qualname__)
        if qualname in _runner.data:
            return fcn(*args, **kwargs)
        if require_fork and multiprocessing.get_start_method() != 'fork':
            raise _unittest.SkipTest('timeout requires unavailable fork interface')
        q = multiprocessing.Queue()
        if multiprocessing.get_start_method() == 'fork':
            _runner.data[q] = (fcn, args, kwargs)
            runner_args = (q, qualname)
        elif args and fcn.__name__.startswith('test') and (_unittest.case.TestCase in args[0].__class__.__mro__):
            runner_args = (q, qualname)
        else:
            runner_args = (q, (qualname, test_timer, args, kwargs))
        test_proc = multiprocessing.Process(target=_runner, args=runner_args)
        test_proc.daemon = True
        try:
            test_proc.start()
        except:
            if type(runner_args[1]) is tuple:
                logging.getLogger(__name__).error("Exception raised spawning timeout subprocess on a platform that does not support 'fork'.  It is likely that either the wrapped function or one of its arguments is not serializable")
            raise
        try:
            resultType, result, stdout = q.get(True, seconds)
        except queue.Empty:
            test_proc.terminate()
            raise timeout_raises('test timed out after %s seconds' % (seconds,)) from None
        finally:
            _runner.data.pop(q, None)
        sys.stdout.write(stdout)
        test_proc.join()
        if resultType == _RunnerResult.call:
            return result
        elif resultType == _RunnerResult.unittest:
            for name, msg in result[0]:
                with args[0].subTest(name):
                    raise args[0].failureException(msg)
            for name, msg in result[1]:
                with args[0].subTest(name):
                    args[0].skipTest(msg)
        else:
            raise result
    return test_timer