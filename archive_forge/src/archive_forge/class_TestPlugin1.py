from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
class TestPlugin1(pybtex.plugin.Plugin):
    pass