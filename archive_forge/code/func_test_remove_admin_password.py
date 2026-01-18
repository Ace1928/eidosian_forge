import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def test_remove_admin_password(self):
    pass_enabled_xml = '<AdminPasswordEnabled>{text}</AdminPasswordEnabled>'
    pass_enabled_true = pass_enabled_xml.format(text='true')
    pass_enabled_false = pass_enabled_xml.format(text='false')
    pass_auto_xml = '<AdminPasswordAuto>{text}</AdminPasswordAuto>'
    pass_auto_true = pass_auto_xml.format(text='true')
    pass_auto_false = pass_auto_xml.format(text='false')
    passwd = '<AdminPassword>testpassword</AdminPassword>'
    assertion_error = False
    for admin_pass_enabled, admin_pass_auto, admin_pass, pass_exists in ((pass_enabled_true, pass_auto_true, passwd, False), (pass_enabled_true, pass_auto_true, '', False), (pass_enabled_true, pass_auto_false, passwd, True), (pass_enabled_true, pass_auto_false, '', False), (pass_enabled_true, '', passwd, False), (pass_enabled_true, '', '', False), (pass_enabled_false, pass_auto_true, passwd, False), (pass_enabled_false, pass_auto_true, '', False), (pass_enabled_false, pass_auto_false, passwd, False), (pass_enabled_false, pass_auto_false, '', False), (pass_enabled_false, '', passwd, False), (pass_enabled_false, '', '', False), ('', pass_auto_true, passwd, False), ('', pass_auto_true, '', False), ('', pass_auto_false, passwd, False), ('', pass_auto_false, '', False), ('', '', passwd, False), ('', '', '', False)):
        try:
            guest_customization_section = ET.fromstring('<GuestCustomizationSection xmlns="http://www.vmware.com/vcloud/v1.5">' + admin_pass_enabled + admin_pass_auto + admin_pass + '</GuestCustomizationSection>')
            self.driver._remove_admin_password(guest_customization_section)
            admin_pass_element = guest_customization_section.find(fixxpath(guest_customization_section, 'AdminPassword'))
            if pass_exists:
                self.assertIsNotNone(admin_pass_element)
            else:
                self.assertIsNone(admin_pass_element)
        except AssertionError:
            assertion_error = True
            print_parameterized_failure([('admin_pass_enabled', admin_pass_enabled), ('admin_pass_auto', admin_pass_auto), ('admin_pass', admin_pass), ('pass_exists', pass_exists)])
    if assertion_error:
        self.fail(msg='Assertion error(s) encountered. Details above.')