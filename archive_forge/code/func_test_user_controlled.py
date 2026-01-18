import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_user_controlled(self):
    self.assertTrue(cfg.Locations.user.is_user_controlled)
    self.assertTrue(cfg.Locations.command_line.is_user_controlled)