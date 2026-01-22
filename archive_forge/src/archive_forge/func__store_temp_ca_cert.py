from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def _store_temp_ca_cert(self):
    if self._cacert:
        try:
            self._cacert_temp_file = tempfile.NamedTemporaryFile()
            self._cacert_temp_file.write(str(self._cacert).encode('utf-8'))
            self._cacert_temp_file.seek(0)
            file_path = self._cacert_temp_file.name
            return file_path
        except Exception:
            LOG.exception('Error when create template file for CA cert')
            if self._cacert_temp_file:
                self._cacert_temp_file.close()
            raise