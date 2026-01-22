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
class _NodeReporter:

    def __init__(self, nodeid: Union[str, TestReport], xml: 'LogXML') -> None:
        self.id = nodeid
        self.xml = xml
        self.add_stats = self.xml.add_stats
        self.family = self.xml.family
        self.duration = 0.0
        self.properties: List[Tuple[str, str]] = []
        self.nodes: List[ET.Element] = []
        self.attrs: Dict[str, str] = {}

    def append(self, node: ET.Element) -> None:
        self.xml.add_stats(node.tag)
        self.nodes.append(node)

    def add_property(self, name: str, value: object) -> None:
        self.properties.append((str(name), bin_xml_escape(value)))

    def add_attribute(self, name: str, value: object) -> None:
        self.attrs[str(name)] = bin_xml_escape(value)

    def make_properties_node(self) -> Optional[ET.Element]:
        """Return a Junit node containing custom properties, if any."""
        if self.properties:
            properties = ET.Element('properties')
            for name, value in self.properties:
                properties.append(ET.Element('property', name=name, value=value))
            return properties
        return None

    def record_testreport(self, testreport: TestReport) -> None:
        names = mangle_test_address(testreport.nodeid)
        existing_attrs = self.attrs
        classnames = names[:-1]
        if self.xml.prefix:
            classnames.insert(0, self.xml.prefix)
        attrs: Dict[str, str] = {'classname': '.'.join(classnames), 'name': bin_xml_escape(names[-1]), 'file': testreport.location[0]}
        if testreport.location[1] is not None:
            attrs['line'] = str(testreport.location[1])
        if hasattr(testreport, 'url'):
            attrs['url'] = testreport.url
        self.attrs = attrs
        self.attrs.update(existing_attrs)
        if self.family == 'xunit1':
            return
        temp_attrs = {}
        for key in self.attrs.keys():
            if key in families[self.family]['testcase']:
                temp_attrs[key] = self.attrs[key]
        self.attrs = temp_attrs

    def to_xml(self) -> ET.Element:
        testcase = ET.Element('testcase', self.attrs, time='%.3f' % self.duration)
        properties = self.make_properties_node()
        if properties is not None:
            testcase.append(properties)
        testcase.extend(self.nodes)
        return testcase

    def _add_simple(self, tag: str, message: str, data: Optional[str]=None) -> None:
        node = ET.Element(tag, message=message)
        node.text = bin_xml_escape(data)
        self.append(node)

    def write_captured_output(self, report: TestReport) -> None:
        if not self.xml.log_passing_tests and report.passed:
            return
        content_out = report.capstdout
        content_log = report.caplog
        content_err = report.capstderr
        if self.xml.logging == 'no':
            return
        content_all = ''
        if self.xml.logging in ['log', 'all']:
            content_all = self._prepare_content(content_log, ' Captured Log ')
        if self.xml.logging in ['system-out', 'out-err', 'all']:
            content_all += self._prepare_content(content_out, ' Captured Out ')
            self._write_content(report, content_all, 'system-out')
            content_all = ''
        if self.xml.logging in ['system-err', 'out-err', 'all']:
            content_all += self._prepare_content(content_err, ' Captured Err ')
            self._write_content(report, content_all, 'system-err')
            content_all = ''
        if content_all:
            self._write_content(report, content_all, 'system-out')

    def _prepare_content(self, content: str, header: str) -> str:
        return '\n'.join([header.center(80, '-'), content, ''])

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)

    def append_pass(self, report: TestReport) -> None:
        self.add_stats('passed')

    def append_failure(self, report: TestReport) -> None:
        if hasattr(report, 'wasxfail'):
            self._add_simple('skipped', 'xfail-marked test passes unexpectedly')
        else:
            assert report.longrepr is not None
            reprcrash: Optional[ReprFileLocation] = getattr(report.longrepr, 'reprcrash', None)
            if reprcrash is not None:
                message = reprcrash.message
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            self._add_simple('failure', message, str(report.longrepr))

    def append_collect_error(self, report: TestReport) -> None:
        assert report.longrepr is not None
        self._add_simple('error', 'collection failure', str(report.longrepr))

    def append_collect_skipped(self, report: TestReport) -> None:
        self._add_simple('skipped', 'collection skipped', str(report.longrepr))

    def append_error(self, report: TestReport) -> None:
        assert report.longrepr is not None
        reprcrash: Optional[ReprFileLocation] = getattr(report.longrepr, 'reprcrash', None)
        if reprcrash is not None:
            reason = reprcrash.message
        else:
            reason = str(report.longrepr)
        if report.when == 'teardown':
            msg = f'failed on teardown with "{reason}"'
        else:
            msg = f'failed on setup with "{reason}"'
        self._add_simple('error', bin_xml_escape(msg), str(report.longrepr))

    def append_skipped(self, report: TestReport) -> None:
        if hasattr(report, 'wasxfail'):
            xfailreason = report.wasxfail
            if xfailreason.startswith('reason: '):
                xfailreason = xfailreason[8:]
            xfailreason = bin_xml_escape(xfailreason)
            skipped = ET.Element('skipped', type='pytest.xfail', message=xfailreason)
            self.append(skipped)
        else:
            assert isinstance(report.longrepr, tuple)
            filename, lineno, skipreason = report.longrepr
            if skipreason.startswith('Skipped: '):
                skipreason = skipreason[9:]
            details = f'{filename}:{lineno}: {skipreason}'
            skipped = ET.Element('skipped', type='pytest.skip', message=bin_xml_escape(skipreason))
            skipped.text = bin_xml_escape(details)
            self.append(skipped)
            self.write_captured_output(report)

    def finalize(self) -> None:
        data = self.to_xml()
        self.__dict__.clear()
        self.to_xml = lambda: data