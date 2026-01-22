import collections
import weakref
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import raw_template_files
class ReadOnlyDict(dict):

    def __setitem__(self, key):
        raise ValueError('Attempted to write to internal TemplateFiles cache')