from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
def _xframe_middleware():
    return 'django.middleware.clickjacking.XFrameOptionsMiddleware' in settings.MIDDLEWARE