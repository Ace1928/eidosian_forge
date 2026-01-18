import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def test_rio_version_hook(self):

    def update_stanza(rev, stanza):
        stanza.add('bla', 'bloe')
    RioVersionInfoBuilder.hooks.install_named_hook('revision', update_stanza, None)
    wt = self.create_branch()
    stanza = self.regen(wt)
    self.assertEqual(['bloe'], stanza.get_all('bla'))