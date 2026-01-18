from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_plugin_class():
    """If a plugin class is passed to find_plugin(), it shoud be returned back."""
    plugin = pybtex.plugin.find_plugin('pybtex.database.input', 'bibtex')
    plugin2 = pybtex.plugin.find_plugin('pybtex.database.input', plugin)
    assert plugin == plugin2