import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def _set_verify(self, headers, value, mask=None, all_bits_masked=False, type_='int'):
    self._verify = {}
    self._verify['headers'] = headers
    self._verify['value'] = value
    self._verify['mask'] = mask
    self._verify['all_bits_masked'] = all_bits_masked
    self._verify['type'] = type_