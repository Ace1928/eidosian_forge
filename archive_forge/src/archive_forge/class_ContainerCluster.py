from typing import List, Optional
from libcloud.common.base import BaseDriver, ConnectionUserAndKey
from libcloud.container.types import ContainerState
class ContainerCluster:
    """
    A cluster group for containers
    """

    def __init__(self, id, name, driver, extra=None):
        """
        :param id: Container Image id.
        :type id: ``str``

        :param name: The name of the image.
        :type  name: ``str``

        :param driver: ContainerDriver instance.
        :type driver: :class:`.ContainerDriver`

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``
        """
        self.id = str(id) if id else None
        self.name = name
        self.driver = driver
        self.extra = extra or {}

    def list_containers(self):
        return self.driver.list_containers(cluster=self)

    def destroy(self):
        return self.driver.destroy_cluster(cluster=self)

    def __repr__(self):
        return '<ContainerCluster: id={}, name={}, provider={} ...>'.format(self.id, self.name, self.driver.name)