from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app.api import appengine_domains_api_client as api_client
from googlecloudsdk.api_lib.run import global_methods as run_methods
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Lists the user's verified domains.