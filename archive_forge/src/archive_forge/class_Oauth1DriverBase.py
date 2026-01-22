import abc
import string
from keystone import exception
class Oauth1DriverBase(object, metaclass=abc.ABCMeta):
    """Interface description for an OAuth1 driver."""

    @abc.abstractmethod
    def create_consumer(self, consumer_ref):
        """Create consumer.

        :param consumer_ref: consumer ref with consumer name
        :type consumer_ref: dict
        :returns: consumer_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def update_consumer(self, consumer_id, consumer_ref):
        """Update consumer.

        :param consumer_id: id of consumer to update
        :type consumer_id: string
        :param consumer_ref: new consumer ref with consumer name
        :type consumer_ref: dict
        :returns: consumer_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_consumers(self):
        """List consumers.

        :returns: list of consumers

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_consumer(self, consumer_id):
        """Get consumer, returns the consumer id (key) and description.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: consumer_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_consumer_with_secret(self, consumer_id):
        """Like get_consumer(), but also returns consumer secret.

        Returned dictionary consumer_ref includes consumer secret.
        Secrets should only be shared upon consumer creation; the
        consumer secret is required to verify incoming OAuth requests.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: consumer_ref containing consumer secret

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_consumer(self, consumer_id):
        """Delete consumer.

        :param consumer_id: id of consumer to get
        :type consumer_id: string
        :returns: None.

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def list_access_tokens(self, user_id):
        """List access tokens.

        :param user_id: search for access tokens authorized by given user id
        :type user_id: string
        :returns: list of access tokens the user has authorized

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def delete_access_token(self, user_id, access_token_id):
        """Delete access token.

        :param user_id: authorizing user id
        :type user_id: string
        :param access_token_id: access token to delete
        :type access_token_id: string
        :returns: None

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def create_request_token(self, consumer_id, requested_project, request_token_duration):
        """Create request token.

        :param consumer_id: the id of the consumer
        :type consumer_id: string
        :param requested_project_id: requested project id
        :type requested_project_id: string
        :param request_token_duration: duration of request token
        :type request_token_duration: string
        :returns: request_token_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_request_token(self, request_token_id):
        """Get request token.

        :param request_token_id: the id of the request token
        :type request_token_id: string
        :returns: request_token_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def get_access_token(self, access_token_id):
        """Get access token.

        :param access_token_id: the id of the access token
        :type access_token_id: string
        :returns: access_token_ref

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def authorize_request_token(self, request_token_id, user_id, role_ids):
        """Authorize request token.

        :param request_token_id: the id of the request token, to be authorized
        :type request_token_id: string
        :param user_id: the id of the authorizing user
        :type user_id: string
        :param role_ids: list of role ids to authorize
        :type role_ids: list
        :returns: verifier

        """
        raise exception.NotImplemented()

    @abc.abstractmethod
    def create_access_token(self, request_id, access_token_duration):
        """Create access token.

        :param request_id: the id of the request token, to be deleted
        :type request_id: string
        :param access_token_duration: duration of an access token
        :type access_token_duration: string
        :returns: access_token_ref

        """
        raise exception.NotImplemented()