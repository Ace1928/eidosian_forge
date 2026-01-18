import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_user_file(self):
    filename = self._write_opt_to_tmp_file('DEFAULT', 'normal_opt', self.id())
    self.conf(['--config-file', filename])
    loc = self.conf.get_location('normal_opt')
    self.assertEqual(cfg.Locations.user, loc.location)
    self.assertEqual(filename, loc.detail)