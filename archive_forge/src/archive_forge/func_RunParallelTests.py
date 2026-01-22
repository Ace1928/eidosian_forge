from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import logging
import os
import subprocess
import re
import sys
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import range
import gslib
from gslib.cloud_api import ProjectIdException
from gslib.command import Command
from gslib.command import ResetFailureCount
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests as tests
from gslib.tests.util import GetTestNames
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import unittest
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def RunParallelTests(self, parallel_integration_tests, max_parallel_tests, coverage_filename):
    """Executes the parallel/isolated portion of the test suite.

    Args:
      parallel_integration_tests: List of tests to execute.
      max_parallel_tests: Maximum number of parallel tests to run at once.
      coverage_filename: If not None, filename for coverage output.

    Returns:
      (int number of test failures, float elapsed time)
    """
    process_list = []
    process_done = []
    process_results = []
    num_parallel_failures = 0
    progress_less_logging_cycles = 0
    completed_as_of_last_log = 0
    num_parallel_tests = len(parallel_integration_tests)
    parallel_start_time = last_log_time = time.time()
    test_index = CreateTestProcesses(parallel_integration_tests, 0, process_list, process_done, max_parallel_tests, root_coverage_file=coverage_filename)
    while len(process_results) < num_parallel_tests:
        for proc_num in range(len(process_list)):
            if process_done[proc_num] or process_list[proc_num].poll() is None:
                continue
            process_done[proc_num] = True
            stdout, stderr = process_list[proc_num].communicate()
            process_list[proc_num].stdout.close()
            process_list[proc_num].stderr.close()
            if process_list[proc_num].returncode != 0:
                num_parallel_failures += 1
            process_results.append(TestProcessData(name=parallel_integration_tests[proc_num], return_code=process_list[proc_num].returncode, stdout=stdout, stderr=stderr))
        if len(process_list) < num_parallel_tests:
            test_index = CreateTestProcesses(parallel_integration_tests, test_index, process_list, process_done, max_parallel_tests, root_coverage_file=coverage_filename)
        if len(process_results) < num_parallel_tests:
            if time.time() - last_log_time > 5:
                print('%d/%d finished - %d failures' % (len(process_results), num_parallel_tests, num_parallel_failures))
                if len(process_results) == completed_as_of_last_log:
                    progress_less_logging_cycles += 1
                else:
                    completed_as_of_last_log = len(process_results)
                    progress_less_logging_cycles = 0
                if progress_less_logging_cycles > 4:
                    still_running = []
                    for proc_num in range(len(process_list)):
                        if not process_done[proc_num]:
                            still_running.append(parallel_integration_tests[proc_num])
                    elapsed = time.time() - parallel_start_time
                    print('{sec} seconds elapsed since beginning parallel tests.\nStill running: {procs}'.format(sec=str(int(elapsed)), procs=still_running))
                last_log_time = time.time()
            time.sleep(1)
    process_run_finish_time = time.time()
    if num_parallel_failures:
        for result in process_results:
            if result.return_code != 0:
                new_stderr = result.stderr.split(b'\n')
                print('Results for failed test %s:' % result.name)
                for line in new_stderr:
                    print(line.decode(UTF8).strip())
    return (num_parallel_failures, process_run_finish_time - parallel_start_time)