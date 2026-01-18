import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def test_privsep_in_loader(self):
    privsep = ['privsep-helper', '--context', 'foo']
    filterlist = wrapper.load_filters([])
    with mock.patch.object(filters.CommandFilter, 'get_exec') as ge:
        ge.return_value = '/fake/privsep-helper'
        filtermatch = wrapper.match_filter(filterlist, privsep)
        self.assertIsNotNone(filtermatch)
        self.assertEqual(['/fake/privsep-helper', '--context', 'foo'], filtermatch.get_command(privsep))