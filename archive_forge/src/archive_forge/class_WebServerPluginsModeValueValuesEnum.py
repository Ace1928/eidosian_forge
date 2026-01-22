from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebServerPluginsModeValueValuesEnum(_messages.Enum):
    """Optional. Whether or not the web server uses custom plugins. If
    unspecified, the field defaults to `PLUGINS_ENABLED`. This field is
    supported for Cloud Composer environments in versions
    composer-3.*.*-airflow-*.*.* and newer.

    Values:
      WEB_SERVER_PLUGINS_MODE_UNSPECIFIED: Default mode.
      PLUGINS_DISABLED: Web server plugins are not supported.
      PLUGINS_ENABLED: Web server plugins are supported.
    """
    WEB_SERVER_PLUGINS_MODE_UNSPECIFIED = 0
    PLUGINS_DISABLED = 1
    PLUGINS_ENABLED = 2