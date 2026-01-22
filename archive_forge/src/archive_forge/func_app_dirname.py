from django.core.exceptions import ImproperlyConfigured, SuspiciousFileOperation
from django.template.utils import get_app_template_dirs
from django.utils._os import safe_join
from django.utils.functional import cached_property
@property
def app_dirname(self):
    raise ImproperlyConfigured("{} doesn't support loading templates from installed applications.".format(self.__class__.__name__))