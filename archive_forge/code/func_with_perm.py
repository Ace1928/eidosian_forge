from django.apps import apps
from django.contrib import auth
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.hashers import make_password
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied
from django.core.mail import send_mail
from django.db import models
from django.db.models.manager import EmptyManager
from django.utils import timezone
from django.utils.itercompat import is_iterable
from django.utils.translation import gettext_lazy as _
from .validators import UnicodeUsernameValidator
def with_perm(self, perm, is_active=True, include_superusers=True, backend=None, obj=None):
    if backend is None:
        backends = auth._get_backends(return_tuples=True)
        if len(backends) == 1:
            backend, _ = backends[0]
        else:
            raise ValueError('You have multiple authentication backends configured and therefore must provide the `backend` argument.')
    elif not isinstance(backend, str):
        raise TypeError('backend must be a dotted import path string (got %r).' % backend)
    else:
        backend = auth.load_backend(backend)
    if hasattr(backend, 'with_perm'):
        return backend.with_perm(perm, is_active=is_active, include_superusers=include_superusers, obj=obj)
    return self.none()