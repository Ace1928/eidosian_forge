from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_register_plugin_1():
    assert pybtex.plugin.register_plugin('pybtex.style.formatting', 'yippikayee', TestPlugin1)
    assert pybtex.plugin.find_plugin('pybtex.style.formatting', 'yippikayee') is TestPlugin1
    assert not pybtex.plugin.register_plugin('pybtex.style.formatting', 'yippikayee', TestPlugin2)
    assert pybtex.plugin.find_plugin('pybtex.style.formatting', 'yippikayee') is TestPlugin1
    assert pybtex.plugin.register_plugin('pybtex.style.formatting', 'yippikayee', TestPlugin2, force=True)
    assert pybtex.plugin.find_plugin('pybtex.style.formatting', 'yippikayee'), TestPlugin2