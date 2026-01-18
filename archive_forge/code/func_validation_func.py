from __future__ import annotations
from fastapi import Depends, Request
from fastapi.background import BackgroundTasks
from lazyops.libs.fastapi_utils.utils import create_function_wrapper
from ..types import errors
from ..types.current_user import CurrentUser, UserRole
from ..types.security import Authorization, APIKey
from ..utils.lazy import logger
from typing import Optional, List, Annotated, Type, Union
def validation_func(*args, **kwargs):
    """
        Validation Function
        """
    current_user = extract_current_user(*args, **kwargs)
    if not current_user.has_user_roles(roles, require_all=require_all):
        if verbose:
            logger.info(f'User {current_user.user_id} does not have required roles: {roles} / {current_user.role}')
        if dry_run:
            return
        raise errors.InvalidRolesException(detail=f'User {current_user.user_id} does not have required roles: {roles}')