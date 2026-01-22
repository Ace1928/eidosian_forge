import io
import logging
import os
import pprint
import sys
import typing as ty
import fixtures
from oslotest import base
import testtools.content
from openstack.tests import fixtures as os_fixtures
from openstack import utils
def add_info_on_exception(self, name, text):

    def add_content(unused):
        self.addDetail(name, testtools.content.text_content(pprint.pformat(text)))
    self.addOnException(add_content)