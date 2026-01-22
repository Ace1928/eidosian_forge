from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def _get_ecr_headers(self, action):
    """
        Get the default headers for a request to the ECR API
        """
    return {'x-amz-target': '{}.{}'.format(ECR_TARGET_BASE, action), 'Content-Type': 'application/x-amz-json-1.1'}