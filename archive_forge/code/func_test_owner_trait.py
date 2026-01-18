import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_owner_trait(self):
    traits = ot.get_traits('OWNER')
    self.assertEqual(['OWNER_CYBORG', 'OWNER_NOVA'], traits)