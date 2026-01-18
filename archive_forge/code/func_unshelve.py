import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def unshelve(self, session, availability_zone=_sentinel, host=None):
    """Unshelve the server.

        :param session: The session to use for making this request.
        :param availability_zone: If specified the instance will be unshelved
            to the availability_zone.
            If None is passed the instance defined availability_zone is unpin
            and the instance will be scheduled to any availability_zone (free
            scheduling).
            If not specified the instance will be unshelved to either its
            defined availability_zone or any availability_zone
            (free scheduling).
        :param host: If specified the host to unshelve the instance.
        """
    data = {}
    if host:
        data['host'] = host
    if availability_zone is None or isinstance(availability_zone, str):
        data['availability_zone'] = availability_zone
    body = {'unshelve': data or None}
    self._action(session, body)