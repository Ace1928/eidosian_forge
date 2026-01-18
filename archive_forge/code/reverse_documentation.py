from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.volumes.replications import client as replications_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp.volumes.replications import flags as replications_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
Reverse a Cloud NetApp Volume Replication's direction in the current project.