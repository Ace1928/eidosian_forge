from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def _to_cluster(self, data):
    return ContainerCluster(id=data['clusterArn'], name=data['clusterName'], driver=self.connection.driver)