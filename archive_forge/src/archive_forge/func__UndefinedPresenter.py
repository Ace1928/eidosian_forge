from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _UndefinedPresenter(dumper, data):
    r = repr(data)
    if r == '[]':
        return dumper.represent_list([])
    if r == '{}':
        return dumper.represent_dict({})
    dumper.represent_undefined(data)