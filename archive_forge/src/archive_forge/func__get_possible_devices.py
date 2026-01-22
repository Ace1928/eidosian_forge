from __future__ import annotations
import os
import typing
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator import linuxfc
from os_brick import utils
def _get_possible_devices(self, hbas: list, targets: list, addressing_mode: Optional[str]=None) -> list:
    """Compute the possible fibre channel device options.

        :param hbas: available hba devices.
        :param targets: tuple of possible wwn addresses and lun combinations.

        :returns: list of (platform, pci_id, wwn, lun) tuples

        Given one or more wwn (mac addresses for fibre channel) ports
        do the matrix math to figure out a set of pci device, wwn
        tuples that are potentially valid (they won't all be). This
        provides a search space for the device connection.

        """
    raw_devices = []
    for hba in hbas:
        platform, pci_num = self._get_pci_num(hba)
        if pci_num is not None:
            for wwn, lun in targets:
                lun = self._linuxscsi.lun_for_addressing(lun, addressing_mode)
                target_wwn = '0x%s' % wwn.lower()
                raw_devices.append((platform, pci_num, target_wwn, lun))
    return raw_devices