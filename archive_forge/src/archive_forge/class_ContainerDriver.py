from typing import List, Optional
from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.container.types import ContainerState
class ContainerDriver(BaseDriver):
    """
    A base ContainerDriver class to derive from

    This class is always subclassed by a specific driver.
    """
    connectionCls = ConnectionUserAndKey
    name = None
    website = None
    supports_clusters = False
    '\n    Whether the driver supports containers being deployed into clusters\n    '

    def __init__(self, key, secret=None, secure=True, host=None, port=None, **kwargs):
        """
        :param    key: API key or username to used (required)
        :type     key: ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :return: ``None``
        """
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, **kwargs)

    def install_image(self, path):
        """
        Install a container image from a remote path.

        :param path: Path to the container image
        :type  path: ``str``

        :rtype: :class:`.ContainerImage`
        """
        raise NotImplementedError('install_image not implemented for this driver')

    def list_images(self):
        """
        List the installed container images

        :rtype: ``list`` of :class:`.ContainerImage`
        """
        raise NotImplementedError('list_images not implemented for this driver')

    def list_containers(self, image=None, cluster=None):
        """
        List the deployed container images

        :param image: Filter to containers with a certain image
        :type  image: :class:`.ContainerImage`

        :param cluster: Filter to containers in a cluster
        :type  cluster: :class:`.ContainerCluster`

        :rtype: ``list`` of :class:`.Container`
        """
        raise NotImplementedError('list_containers not implemented for this driver')

    def deploy_container(self, name, image, cluster=None, parameters=None, start=True):
        """
        Deploy an installed container image

        :param name: The name of the new container
        :type  name: ``str``

        :param image: The container image to deploy
        :type  image: :class:`.ContainerImage`

        :param cluster: The cluster to deploy to, None is default
        :type  cluster: :class:`.ContainerCluster`

        :param parameters: Container Image parameters
        :type  parameters: ``str``

        :param start: Start the container on deployment
        :type  start: ``bool``

        :rtype: :class:`.Container`
        """
        raise NotImplementedError('deploy_container not implemented for this driver')

    def get_container(self, id):
        """
        Get a container by ID

        :param id: The ID of the container to get
        :type  id: ``str``

        :rtype: :class:`.Container`
        """
        raise NotImplementedError('get_container not implemented for this driver')

    def start_container(self, container):
        """
        Start a deployed container

        :param container: The container to start
        :type  container: :class:`.Container`

        :rtype: :class:`.Container`
        """
        raise NotImplementedError('start_container not implemented for this driver')

    def stop_container(self, container):
        """
        Stop a deployed container

        :param container: The container to stop
        :type  container: :class:`.Container`

        :rtype: :class:`.Container`
        """
        raise NotImplementedError('stop_container not implemented for this driver')

    def restart_container(self, container):
        """
        Restart a deployed container

        :param container: The container to restart
        :type  container: :class:`.Container`

        :rtype: :class:`.Container`
        """
        raise NotImplementedError('restart_container not implemented for this driver')

    def destroy_container(self, container):
        """
        Destroy a deployed container

        :param container: The container to destroy
        :type  container: :class:`.Container`

        :rtype: ``bool``
        """
        raise NotImplementedError('destroy_container not implemented for this driver')

    def list_locations(self):
        """
        Get a list of potential locations to deploy clusters into

        :rtype: ``list`` of :class:`.ClusterLocation`
        """
        raise NotImplementedError('list_locations not implemented for this driver')

    def create_cluster(self, name, location=None):
        """
        Create a container cluster

        :param  name: The name of the cluster
        :type   name: ``str``

        :param  location: The location to create the cluster in
        :type   location: :class:`.ClusterLocation`

        :rtype: :class:`.ContainerCluster`
        """
        raise NotImplementedError('create_cluster not implemented for this driver')

    def destroy_cluster(self, cluster):
        """
        Delete a cluster

        :return: ``True`` if the destroy was successful, otherwise ``False``.
        :rtype: ``bool``
        """
        raise NotImplementedError('destroy_cluster not implemented for this driver')

    def list_clusters(self, location=None):
        """
        Get a list of potential locations to deploy clusters into

        :param  location: The location to search in
        :type   location: :class:`.ClusterLocation`

        :rtype: ``list`` of :class:`.ContainerCluster`
        """
        raise NotImplementedError('list_clusters not implemented for this driver')

    def get_cluster(self, id):
        """
        Get a cluster by ID

        :param id: The ID of the cluster to get
        :type  id: ``str``

        :rtype: :class:`.ContainerCluster`
        """
        raise NotImplementedError('list_clusters not implemented for this driver')