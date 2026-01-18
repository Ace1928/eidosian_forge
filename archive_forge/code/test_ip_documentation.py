from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_firewall_api_client as api_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import firewall_rules_util
from googlecloudsdk.core import log
Display firewall rules that match a given IP.