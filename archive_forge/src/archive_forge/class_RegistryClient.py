from base64 import b64encode
from libcloud.common.base import Connection, JsonResponse
from libcloud.container.base import ContainerImage
class RegistryClient:
    """
    A client for the Docker v2 registry API
    """
    connectionCls = DockerHubConnection

    def __init__(self, host, username=None, password=None, **kwargs):
        """
        Construct a Docker registry client

        :param host: Your registry endpoint, e.g. 'registry.hub.docker.com'
        :type  host: ``str``

        :param username: (optional) Your registry account username
        :type  username: ``str``

        :param password: (optional) Your registry account password
        :type  password: ``str``
        """
        self.connection = self.connectionCls(host, username, password, **kwargs)

    def list_images(self, repository_name, namespace='library', max_count=100):
        """
        List the tags (versions) in a repository

        :param  repository_name: The name of the repository e.g. 'ubuntu'
        :type   repository_name: ``str``

        :param  namespace: (optional) The docker namespace
        :type   namespace: ``str``

        :param  max_count: The maximum number of records to return
        :type   max_count: ``int``

        :return: A list of images
        :rtype: ``list`` of :class:`libcloud.container.base.ContainerImage`
        """
        path = '/v2/repositories/{}/{}/tags/?page=1&page_size={}'.format(namespace, repository_name, max_count)
        response = self.connection.request(path)
        images = []
        for image in response.object['results']:
            images.append(self._to_image(repository_name, image))
        return images

    def get_repository(self, repository_name, namespace='library'):
        """
        Get the information about a specific repository

        :param  repository_name: The name of the repository e.g. 'ubuntu'
        :type   repository_name: ``str``

        :param  namespace: (optional) The docker namespace
        :type   namespace: ``str``

        :return: The details of the repository
        :rtype: ``object``
        """
        path = '/v2/repositories/{}/{}/'.format(namespace, repository_name)
        response = self.connection.request(path)
        return response.object

    def get_image(self, repository_name, tag='latest', namespace='library'):
        """
        Get an image from a repository with a specific tag

        :param repository_name: The name of the repository, e.g. ubuntu
        :type  repository_name: ``str``

        :param  tag: (optional) The image tag (defaults to latest)
        :type   tag: ``str``

        :param  namespace: (optional) The docker namespace
        :type   namespace: ``str``

        :return: A container image
        :rtype: :class:`libcloud.container.base.ContainerImage`
        """
        path = '/v2/repositories/{}/{}/tags/{}/'.format(namespace, repository_name, tag)
        response = self.connection.request(path)
        return self._to_image(repository_name, response.object)

    def _to_image(self, repository_name, obj):
        path = '{}/{}:{}'.format(self.connection.host, repository_name, obj['name'])
        return ContainerImage(id=obj['id'], path=path, name=path, version=obj['name'], extra={'full_size': obj['full_size']}, driver=None)