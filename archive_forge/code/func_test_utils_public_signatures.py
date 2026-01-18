import inspect
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import rdpconsoleutils
from os_win.utils.compute import vmutils
from os_win.utils.dns import dnsutils
from os_win.utils import hostutils
from os_win.utils.io import ioutils
from os_win.utils.network import networkutils
from os_win.utils import pathutils
from os_win.utils import processutils
from os_win.utils.storage import diskutils
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.storage import smbutils
from os_win.utils.storage.virtdisk import vhdutils
from os_win import utilsfactory
def test_utils_public_signatures(self):
    for module_name in utilsfactory.utils_map.keys():
        classes = utilsfactory.utils_map[module_name]
        if len(classes) < 2:
            continue
        base_class_dict = classes[0]
        base_class = importutils.import_object(base_class_dict['path'])
        for i in range(1, len(classes)):
            tested_class_dict = classes[i]
            tested_class = importutils.import_object(tested_class_dict['path'])
            self.assertPublicAPISignatures(base_class, tested_class)
            self.assertPublicAPISignatures(tested_class, base_class)