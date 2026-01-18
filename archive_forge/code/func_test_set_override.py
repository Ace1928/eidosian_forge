import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_set_override(self):
    self.conf.set_override('normal_opt', self.id())
    self.conf([])
    loc = self.conf.get_location('normal_opt')
    self.assertEqual(cfg.Locations.set_override, loc.location)
    self.assertIn('test_get_location.py', loc.detail)