import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def _dump_timings(self, timings):

    class Tyme(object):

        def __init__(self, url, seconds):
            self.url = url
            self.seconds = seconds
    results = [Tyme(url, end - start) for url, start, end in timings]
    total = 0.0
    for tyme in results:
        total += tyme.seconds
    results.append(Tyme('Total', total))
    cliutils.print_list(results, ['url', 'seconds'], sortby_index=None)