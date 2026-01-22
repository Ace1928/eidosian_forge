import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def add_trait(self, session, trait):
    """Add a trait to the node.

        :param session: The session to use for making this request.
        :param trait: The trait to add to the node.
        :returns: ``None``
        """
    session = self._get_session(session)
    version = utils.pick_microversion(session, '1.37')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'traits', trait)
    response = session.put(request.url, json=None, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to add trait {trait} for node {node}'.format(trait=trait, node=self.id)
    exceptions.raise_from_response(response, error_message=msg)
    self.traits = list(set(self.traits or ()) | {trait})