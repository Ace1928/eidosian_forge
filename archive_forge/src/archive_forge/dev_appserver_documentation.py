from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
from bootstrapping import bootstrapping
from googlecloudsdk.api_lib.app import wrapper_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.emulators import datastore_util
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import metrics
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import platforms
Launches dev_appserver.py.