import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
class GetFile(function.Function):
    """A function for including a file inline.

    Takes the form::

        get_file: <file_key>

    And resolves to the content stored in the files dictionary under the given
    key.
    """

    def __init__(self, stack, fn_name, args):
        super(GetFile, self).__init__(stack, fn_name, args)
        self.files = self.stack.t.files if self.stack is not None else None

    def result(self):
        assert self.files is not None, 'No stack definition in Function'
        args = function.resolve(self.args)
        if not isinstance(args, str):
            raise TypeError(_('Argument to "%s" must be a string') % self.fn_name)
        f = self.files.get(args)
        if f is None:
            fmt_data = {'fn_name': self.fn_name, 'file_key': args}
            raise ValueError(_('No content found in the "files" section for %(fn_name)s path: %(file_key)s') % fmt_data)
        return f