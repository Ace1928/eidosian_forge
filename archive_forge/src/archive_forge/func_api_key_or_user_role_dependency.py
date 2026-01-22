from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def api_key_or_user_role_dependency(api_keys: Union[str, List[str]], role: Optional[Union[str, UserRole]]=None, dry_run: Optional[bool]=False, verbose: Optional[bool]=False):
    """
    Creates an api key validator wrapper
    """
    if not isinstance(api_keys, list):
        api_keys = [api_keys]
    user_role = UserRole.parse_role(role) if role else None

    async def has_api_key_or_role(current_user: OptionalUser, api_key: Optional[APIKey]=None):
        """
        Checks if the api key is valid
        """
        if not api_key and (not current_user):
            if verbose:
                logger.info('No api key or user found')
            raise errors.NoAPIKeyException()
        if api_key and api_key in api_keys:
            return
        if current_user and current_user.is_valid:
            if user_role and current_user.role < user_role:
                if verbose:
                    logger.info(f'User {current_user.user_id} does not have required role: {user_role}')
                if dry_run:
                    return
                raise errors.InvalidAuthRoleException(detail=f'User {current_user.user_id} does not have required role: {user_role}')
            return
        if verbose:
            logger.info(f'`{api_key}` is not a valid api key')
        if dry_run:
            return
        raise errors.InvalidAPIKeyException(detail=f'`{api_key}` is not a valid api key')
    return Depends(has_api_key_or_role)