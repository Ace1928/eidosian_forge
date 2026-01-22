import abc
from keystone import exception
class ShadowUsersDriverBase(object, metaclass=abc.ABCMeta):
    """Interface description for an Shadow Users driver."""

    @abc.abstractmethod
    def create_federated_object(self, fed_dict):
        """Create a new federated object.

        :param dict federated_dict: Reference to the federated user
        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def create_federated_user(self, domain_id, federated_dict, email=None):
        """Create a new user with the federated identity.

        :param domain_id: The domain ID of the IdP used for the federated user
        :param dict federated_dict: Reference to the federated user
        :param email: Federated user's email
        :returns dict: Containing the user reference

        """
        raise exception.NotImplemented()

    def delete_federated_object(self, user_id):
        """Delete a user's federated objects.

        :param user_id: Unique identifier of the user
        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_federated_objects(self, user_id):
        """Get all federated objects for a user.

        :param user_id: Unique identifier of the user
        :returns list: Containing the user's federated objects

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_federated_user(self, idp_id, protocol_id, unique_id):
        """Return the found user for the federated identity.

        :param idp_id: The identity provider ID
        :param protocol_id: The federation protocol ID
        :param unique_id: The unique ID for the user
        :returns dict: Containing the user reference

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def update_federated_user_display_name(self, idp_id, protocol_id, unique_id, display_name):
        """Update federated user's display name if changed.

        :param idp_id: The identity provider ID
        :param protocol_id: The federation protocol ID
        :param unique_id: The unique ID for the user
        :param display_name: The user's display name

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_user(self, user_id):
        """Return the found user.

        :param user_id: Unique identifier of the user
        :returns dict: Containing the user reference

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def create_nonlocal_user(self, user_dict):
        """Create a new non-local user.

        :param dict user_dict: Reference to the non-local user
        :returns dict: Containing the user reference

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def set_last_active_at(self, user_id):
        """Set the last active at date for the user.

        :param user_id: Unique identifier of the user

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_federated_users_info(self, hints=None):
        """Get the shadow users info with the specified filters.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.
        :returns list: A list of objects that containing the shadow users
                       reference.

        """
        raise exception.NotImplemented()