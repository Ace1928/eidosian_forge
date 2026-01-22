from openstack.clustering.v1 import action as _action
from openstack import exceptions
from openstack import resource
def _delete_response(self, response, error_message=None):
    exceptions.raise_from_response(response, error_message=error_message)
    location = response.headers['Location']
    action_id = location.split('/')[-1]
    action = _action.Action.existing(id=action_id, connection=self._connection)
    return action