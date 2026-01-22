from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dns import util as dns_api_util
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import six
class DnsUpdateMask(object):
    """Class with information which parts of dns_settings should be updated."""

    def __init__(self, name_servers=False, glue_records=False, google_domains_dnssec=False, custom_dnssec=False):
        self.name_servers = name_servers
        self.glue_records = glue_records
        self.google_domains_dnssec = google_domains_dnssec
        self.custom_dnssec = custom_dnssec