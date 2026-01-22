from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import tempfile
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import transfer
from googlecloudsdk.api_lib.lifesciences import exceptions as lifesciences_exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
import six
def GetLifeSciencesMessages(version='v2beta'):
    return core_apis.GetMessagesModule('lifesciences', version)