from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import csv
import datetime
import enum
import os
from googlecloudsdk.command_lib.storage import thread_messages
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
Writes data to manifest file.