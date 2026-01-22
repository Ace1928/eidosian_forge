from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.krmapihosting import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def LocationAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='location', help_text="The name of the Config Controller instance location. Currently, only ``us-central1'', ``us-east1'', ``us-east4'', ``us-east5'', ``us-west2'', ``northamerica-northeast1'', ``northamerica-northeast2'', ``europe-north1'', ``europe-west1'', ``europe-west3'', ``europe-west6'', ``australia-southeast1'', ``australia-southeast2'', ``asia-northeast1'', ``asia-northeast2'' and ``asia-southeast1'' are supported.")