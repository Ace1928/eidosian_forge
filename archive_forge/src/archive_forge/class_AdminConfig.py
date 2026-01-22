from django.apps import AppConfig
from django.contrib.admin.checks import check_admin_app, check_dependencies
from django.core import checks
from django.utils.translation import gettext_lazy as _
class AdminConfig(SimpleAdminConfig):
    """The default AppConfig for admin which does autodiscovery."""
    default = True

    def ready(self):
        super().ready()
        self.module.autodiscover()