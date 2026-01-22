from typing import Union
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.connectgateway.v1alpha1 import connectgateway_v1alpha1_messages as messages_v1alpha1
class TYPES:
    GenerateCredentialsResponse = Union[messages_v1alpha1.GenerateCredentialsResponse]