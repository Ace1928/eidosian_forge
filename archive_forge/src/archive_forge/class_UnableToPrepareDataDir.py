from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import tempfile
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
class UnableToPrepareDataDir(exceptions.Error):

    def __init__(self):
        super(UnableToPrepareDataDir, self).__init__('Unable to prepare the data directory for the emulator')