import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
def _check_combinations(tag):
    self.repo.commit()
    self.assertEqual(dict(), get_kwargs(tag))
    self.repo.commit('sem-ver: bugfix')
    self.assertEqual(dict(), get_kwargs(tag))
    self.repo.commit('sem-ver: feature')
    self.assertEqual(dict(minor=True), get_kwargs(tag))
    self.repo.uncommit()
    self.repo.commit('sem-ver: deprecation')
    self.assertEqual(dict(minor=True), get_kwargs(tag))
    self.repo.uncommit()
    self.repo.commit('sem-ver: api-break')
    self.assertEqual(dict(major=True), get_kwargs(tag))
    self.repo.commit('sem-ver: deprecation')
    self.assertEqual(dict(major=True, minor=True), get_kwargs(tag))