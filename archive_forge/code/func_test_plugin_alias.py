from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_plugin_alias():
    pybtex.plugin._DEFAULT_PLUGINS['pybtex.legacy.input'] = 'punchcard'
    assert pybtex.plugin.register_plugin('pybtex.legacy.input', 'punchcard', TestPlugin4)
    assert pybtex.plugin.register_plugin('pybtex.legacy.input.aliases', 'punchedcard', TestPlugin4)
    assert list(pybtex.plugin.enumerate_plugin_names('pybtex.legacy.input')) == ['punchcard']
    plugin = pybtex.plugin.find_plugin('pybtex.legacy.input', 'punchedcard')
    assert plugin is TestPlugin4
    del pybtex.plugin._DEFAULT_PLUGINS['pybtex.legacy.input']