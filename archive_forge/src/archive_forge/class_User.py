from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
class User(SaveMixin, ObjectDeleteMixin, RESTObject):
    _repr_attr = 'username'
    customattributes: UserCustomAttributeManager
    emails: 'UserEmailManager'
    events: UserEventManager
    followers_users: 'UserFollowersManager'
    following_users: 'UserFollowingManager'
    gpgkeys: 'UserGPGKeyManager'
    identityproviders: 'UserIdentityProviderManager'
    impersonationtokens: 'UserImpersonationTokenManager'
    keys: 'UserKeyManager'
    memberships: 'UserMembershipManager'
    personal_access_tokens: UserPersonalAccessTokenManager
    projects: 'UserProjectManager'
    starred_projects: 'StarredProjectManager'
    status: 'UserStatusManager'

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabBlockError)
    def block(self, **kwargs: Any) -> Optional[bool]:
        """Block the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabBlockError: If the user could not be blocked

        Returns:
            Whether the user status has been changed
        """
        path = f'/users/{self.encoded_id}/block'
        server_data = cast(Optional[bool], self.manager.gitlab.http_post(path, **kwargs))
        if server_data is True:
            self._attrs['state'] = 'blocked'
        return server_data

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabFollowError)
    def follow(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Follow the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabFollowError: If the user could not be followed

        Returns:
            The new object data (*not* a RESTObject)
        """
        path = f'/users/{self.encoded_id}/follow'
        return self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabUnfollowError)
    def unfollow(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Unfollow the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUnfollowError: If the user could not be followed

        Returns:
            The new object data (*not* a RESTObject)
        """
        path = f'/users/{self.encoded_id}/unfollow'
        return self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabUnblockError)
    def unblock(self, **kwargs: Any) -> Optional[bool]:
        """Unblock the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUnblockError: If the user could not be unblocked

        Returns:
            Whether the user status has been changed
        """
        path = f'/users/{self.encoded_id}/unblock'
        server_data = cast(Optional[bool], self.manager.gitlab.http_post(path, **kwargs))
        if server_data is True:
            self._attrs['state'] = 'active'
        return server_data

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabDeactivateError)
    def deactivate(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Deactivate the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeactivateError: If the user could not be deactivated

        Returns:
            Whether the user status has been changed
        """
        path = f'/users/{self.encoded_id}/deactivate'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if server_data:
            self._attrs['state'] = 'deactivated'
        return server_data

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabActivateError)
    def activate(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Activate the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabActivateError: If the user could not be activated

        Returns:
            Whether the user status has been changed
        """
        path = f'/users/{self.encoded_id}/activate'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if server_data:
            self._attrs['state'] = 'active'
        return server_data

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabUserApproveError)
    def approve(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Approve a user creation request.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUserApproveError: If the user could not be activated

        Returns:
            The new object data (*not* a RESTObject)
        """
        path = f'/users/{self.encoded_id}/approve'
        return self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabUserRejectError)
    def reject(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Reject a user creation request.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUserRejectError: If the user could not be rejected

        Returns:
            The new object data (*not* a RESTObject)
        """
        path = f'/users/{self.encoded_id}/reject'
        return self.manager.gitlab.http_post(path, **kwargs)

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabBanError)
    def ban(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Ban the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabBanError: If the user could not be banned

        Returns:
            Whether the user has been banned
        """
        path = f'/users/{self.encoded_id}/ban'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if server_data:
            self._attrs['state'] = 'banned'
        return server_data

    @cli.register_custom_action('User')
    @exc.on_http_error(exc.GitlabUnbanError)
    def unban(self, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
        """Unban the user.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUnbanError: If the user could not be unbanned

        Returns:
            Whether the user has been unbanned
        """
        path = f'/users/{self.encoded_id}/unban'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if server_data:
            self._attrs['state'] = 'active'
        return server_data