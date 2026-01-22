import os
from mako.cache import CacheImpl
from mako.cache import register_plugin
from mako.template import Template
from .assertions import eq_
from .config import config
def _do_file_test(self, filename, expected, filters=None, unicode_=True, template_args=None, **kw):
    t1 = self._file_template(filename, **kw)
    self._do_test(t1, expected, filters=filters, unicode_=unicode_, template_args=template_args)