import inspect
import re
from asgiref.sync import sync_to_async
from django.apps import apps as django_apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.middleware.csrf import rotate_token
from django.utils.crypto import constant_time_compare
from django.utils.module_loading import import_string
from django.views.decorators.debug import sensitive_variables
from .signals import user_logged_in, user_logged_out, user_login_failed
def _get_user_session_key(request):
    return get_user_model()._meta.pk.to_python(request.session[SESSION_KEY])