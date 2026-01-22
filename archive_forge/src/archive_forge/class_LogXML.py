from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
class LogXML:

    def __init__(self, logfile, prefix: Optional[str], suite_name: str='pytest', logging: str='no', report_duration: str='total', family='xunit1', log_passing_tests: bool=True) -> None:
        logfile = os.path.expanduser(os.path.expandvars(logfile))
        self.logfile = os.path.normpath(os.path.abspath(logfile))
        self.prefix = prefix
        self.suite_name = suite_name
        self.logging = logging
        self.log_passing_tests = log_passing_tests
        self.report_duration = report_duration
        self.family = family
        self.stats: Dict[str, int] = dict.fromkeys(['error', 'passed', 'failure', 'skipped'], 0)
        self.node_reporters: Dict[Tuple[Union[str, TestReport], object], _NodeReporter] = {}
        self.node_reporters_ordered: List[_NodeReporter] = []
        self.global_properties: List[Tuple[str, str]] = []
        self.open_reports: List[TestReport] = []
        self.cnt_double_fail_tests = 0
        if self.family == 'legacy':
            self.family = 'xunit1'

    def finalize(self, report: TestReport) -> None:
        nodeid = getattr(report, 'nodeid', report)
        workernode = getattr(report, 'node', None)
        reporter = self.node_reporters.pop((nodeid, workernode))
        for propname, propvalue in report.user_properties:
            reporter.add_property(propname, str(propvalue))
        if reporter is not None:
            reporter.finalize()

    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporter:
        nodeid: Union[str, TestReport] = getattr(report, 'nodeid', report)
        workernode = getattr(report, 'node', None)
        key = (nodeid, workernode)
        if key in self.node_reporters:
            return self.node_reporters[key]
        reporter = _NodeReporter(nodeid, self)
        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)
        return reporter

    def add_stats(self, key: str) -> None:
        if key in self.stats:
            self.stats[key] += 1

    def _opentestcase(self, report: TestReport) -> _NodeReporter:
        reporter = self.node_reporter(report)
        reporter.record_testreport(report)
        return reporter

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """Handle a setup/call/teardown report, generating the appropriate
        XML tags as necessary.

        Note: due to plugins like xdist, this hook may be called in interlaced
        order with reports from other nodes. For example:

        Usual call order:
            -> setup node1
            -> call node1
            -> teardown node1
            -> setup node2
            -> call node2
            -> teardown node2

        Possible call order in xdist:
            -> setup node1
            -> call node1
            -> setup node2
            -> call node2
            -> teardown node2
            -> teardown node1
        """
        close_report = None
        if report.passed:
            if report.when == 'call':
                reporter = self._opentestcase(report)
                reporter.append_pass(report)
        elif report.failed:
            if report.when == 'teardown':
                report_wid = getattr(report, 'worker_id', None)
                report_ii = getattr(report, 'item_index', None)
                close_report = next((rep for rep in self.open_reports if rep.nodeid == report.nodeid and getattr(rep, 'item_index', None) == report_ii and (getattr(rep, 'worker_id', None) == report_wid)), None)
                if close_report:
                    self.finalize(close_report)
                    self.cnt_double_fail_tests += 1
            reporter = self._opentestcase(report)
            if report.when == 'call':
                reporter.append_failure(report)
                self.open_reports.append(report)
                if not self.log_passing_tests:
                    reporter.write_captured_output(report)
            else:
                reporter.append_error(report)
        elif report.skipped:
            reporter = self._opentestcase(report)
            reporter.append_skipped(report)
        self.update_testcase_duration(report)
        if report.when == 'teardown':
            reporter = self._opentestcase(report)
            reporter.write_captured_output(report)
            self.finalize(report)
            report_wid = getattr(report, 'worker_id', None)
            report_ii = getattr(report, 'item_index', None)
            close_report = next((rep for rep in self.open_reports if rep.nodeid == report.nodeid and getattr(rep, 'item_index', None) == report_ii and (getattr(rep, 'worker_id', None) == report_wid)), None)
            if close_report:
                self.open_reports.remove(close_report)

    def update_testcase_duration(self, report: TestReport) -> None:
        """Accumulate total duration for nodeid from given report and update
        the Junit.testcase with the new total if already created."""
        if self.report_duration in {'total', report.when}:
            reporter = self.node_reporter(report)
            reporter.duration += getattr(report, 'duration', 0.0)

    def pytest_collectreport(self, report: TestReport) -> None:
        if not report.passed:
            reporter = self._opentestcase(report)
            if report.failed:
                reporter.append_collect_error(report)
            else:
                reporter.append_collect_skipped(report)

    def pytest_internalerror(self, excrepr: ExceptionRepr) -> None:
        reporter = self.node_reporter('internal')
        reporter.attrs.update(classname='pytest', name='internal')
        reporter._add_simple('error', 'internal error', str(excrepr))

    def pytest_sessionstart(self) -> None:
        self.suite_start_time = timing.time()

    def pytest_sessionfinish(self) -> None:
        dirname = os.path.dirname(os.path.abspath(self.logfile))
        os.makedirs(dirname, exist_ok=True)
        with open(self.logfile, 'w', encoding='utf-8') as logfile:
            suite_stop_time = timing.time()
            suite_time_delta = suite_stop_time - self.suite_start_time
            numtests = self.stats['passed'] + self.stats['failure'] + self.stats['skipped'] + self.stats['error'] - self.cnt_double_fail_tests
            logfile.write('<?xml version="1.0" encoding="utf-8"?>')
            suite_node = ET.Element('testsuite', name=self.suite_name, errors=str(self.stats['error']), failures=str(self.stats['failure']), skipped=str(self.stats['skipped']), tests=str(numtests), time='%.3f' % suite_time_delta, timestamp=datetime.fromtimestamp(self.suite_start_time).isoformat(), hostname=platform.node())
            global_properties = self._get_global_properties_node()
            if global_properties is not None:
                suite_node.append(global_properties)
            for node_reporter in self.node_reporters_ordered:
                suite_node.append(node_reporter.to_xml())
            testsuites = ET.Element('testsuites')
            testsuites.append(suite_node)
            logfile.write(ET.tostring(testsuites, encoding='unicode'))

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        terminalreporter.write_sep('-', f'generated xml file: {self.logfile}')

    def add_global_property(self, name: str, value: object) -> None:
        __tracebackhide__ = True
        _check_record_param_type('name', name)
        self.global_properties.append((name, bin_xml_escape(value)))

    def _get_global_properties_node(self) -> Optional[ET.Element]:
        """Return a Junit node containing custom properties, if any."""
        if self.global_properties:
            properties = ET.Element('properties')
            for name, value in self.global_properties:
                properties.append(ET.Element('property', name=name, value=value))
            return properties
        return None