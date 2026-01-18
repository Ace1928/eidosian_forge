from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_register_plugin_2():
    assert not pybtex.plugin.register_plugin('pybtex.style.formatting', 'plain', TestPlugin2)
    plugin = pybtex.plugin.find_plugin('pybtex.style.formatting', 'plain')
    assert plugin is not TestPlugin2
    assert plugin is pybtex.style.formatting.plain.Style