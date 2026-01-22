from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
from collections import defaultdict
from collections import namedtuple
import contextlib
import datetime
import json
import logging
import math
import multiprocessing
import os
import random
import re
import socket
import string
import subprocess
import tempfile
import time
import boto
import boto.gs.connection
import six
from six.moves import cStringIO
from six.moves import http_client
from six.moves import xrange
from six.moves import range
import gslib
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DummyArgChecker
from gslib.command_argument import CommandArgument
from gslib.commands import config
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.file_part import FilePart
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetDiskCounters
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IsRunningInCiEnvironment
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeBitsHumanReadable
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import Percentile
def _GetDiskCounters(self):
    """Retrieves disk I/O statistics for all disks.

    Adapted from the psutil module's psutil._pslinux.disk_io_counters:
      http://code.google.com/p/psutil/source/browse/trunk/psutil/_pslinux.py

    Originally distributed under under a BSD license.
    Original Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola.

    Returns:
      A dictionary containing disk names mapped to the disk counters from
      /disk/diskstats.
    """
    sector_size = 512
    partitions = []
    with open('/proc/partitions', 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            _, _, _, name = line.split()
            if name[-1].isdigit():
                partitions.append(name)
    retdict = {}
    with open('/proc/diskstats', 'r') as f:
        for line in f:
            values = line.split()[:11]
            _, _, name, reads, _, rbytes, rtime, writes, _, wbytes, wtime = values
            if name in partitions:
                rbytes = int(rbytes) * sector_size
                wbytes = int(wbytes) * sector_size
                reads = int(reads)
                writes = int(writes)
                rtime = int(rtime)
                wtime = int(wtime)
                retdict[name] = (reads, writes, rbytes, wbytes, rtime, wtime)
    return retdict