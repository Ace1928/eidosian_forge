from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def PipelineToPipelineRef(pipeline):
    pipeline_ref = resources.REGISTRY.ParseRelativeName(pipeline.name, collection='clouddeploy.projects.locations.deliveryPipelines')
    return pipeline_ref