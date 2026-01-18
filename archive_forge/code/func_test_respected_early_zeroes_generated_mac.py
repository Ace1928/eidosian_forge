import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
@mock.patch.object(random, 'getrandbits', return_value=162)
def test_respected_early_zeroes_generated_mac(self, mock_rnd):
    mac1 = ['00', 'bb', 'cc', '00', 'ee', 'ff']
    generator = itertools.islice(net.random_mac_generator(mac1), 1)
    self.assertEqual(['00:bb:cc:a2:a2:a2'], list(generator))
    mac2 = ['aa', '00', 'cc', '00', 'ee', 'ff']
    generator = itertools.islice(net.random_mac_generator(mac2), 1)
    self.assertEqual(['aa:00:cc:a2:a2:a2'], list(generator))
    mac3 = ['aa', 'bb', '00', '00', 'ee', 'ff']
    generator = itertools.islice(net.random_mac_generator(mac3), 1)
    self.assertEqual(['aa:bb:00:a2:a2:a2'], list(generator))
    mock_rnd.assert_called_with(8)