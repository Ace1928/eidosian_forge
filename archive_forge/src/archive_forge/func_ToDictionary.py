from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
def ToDictionary(self):
    w = DictionaryWriter(self)
    w.Write('last_update_check_time')
    w.Write('last_update_check_revision')
    w.WriteList('notifications', func=NotificationSpec.ToDictionary)
    w.WriteDict('last_nag_times')
    return w.Dictionary()