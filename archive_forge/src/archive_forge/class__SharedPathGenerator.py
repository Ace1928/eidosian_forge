from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib import genomics as lib
from googlecloudsdk.api_lib.genomics import exceptions
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class _SharedPathGenerator(object):

    def __init__(self, root):
        self.root = root
        self.index = -1

    def Generate(self):
        self.index += 1
        return '/%s/%s%d' % (SHARED_DISK, self.root, self.index)