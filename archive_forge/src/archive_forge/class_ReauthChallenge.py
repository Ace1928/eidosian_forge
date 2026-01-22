import abc
import base64
import sys
import pyu2f.convenience.authenticator
import pyu2f.errors
import pyu2f.model
import six
from google_reauth import _helpers, errors
@six.add_metaclass(abc.ABCMeta)
class ReauthChallenge(object):
    """Base class for reauth challenges."""

    @property
    @abc.abstractmethod
    def name(self):
        """Returns the name of the challenge."""
        pass

    @property
    @abc.abstractmethod
    def is_locally_eligible(self):
        """Returns true if a challenge is supported locally on this machine."""
        pass

    @abc.abstractmethod
    def obtain_challenge_input(self, metadata):
        """Performs logic required to obtain credentials and returns it.

        Args:
            metadata: challenge metadata returned in the 'challenges' field in
                the initial reauth request. Includes the 'challengeType' field
                and other challenge-specific fields.

        Returns:
            response that will be send to the reauth service as the content of
            the 'proposalResponse' field in the request body. Usually a dict
            with the keys specific to the challenge. For example,
            {'credential': password} for password challenge.
        """
        pass