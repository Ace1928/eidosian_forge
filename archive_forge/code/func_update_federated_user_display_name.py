import abc
from keystone import exception
@abc.abstractmethod
def update_federated_user_display_name(self, idp_id, protocol_id, unique_id, display_name):
    """Update federated user's display name if changed.

        :param idp_id: The identity provider ID
        :param protocol_id: The federation protocol ID
        :param unique_id: The unique ID for the user
        :param display_name: The user's display name

        """
    raise exception.NotImplemented()