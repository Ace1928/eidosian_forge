from __future__ import annotations
import copy
from typing import Any, Generator  # noqa: H301
from os_brick.initiator import initiator_connector
class BaseISCSIConnector(initiator_connector.InitiatorConnector):

    def _iterate_all_targets(self, connection_properties: dict) -> Generator[dict[str, Any], None, None]:
        for portal, iqn, lun in self._get_all_targets(connection_properties):
            props = copy.deepcopy(connection_properties)
            props['target_portal'] = portal
            props['target_iqn'] = iqn
            props['target_lun'] = lun
            for key in ('target_portals', 'target_iqns', 'target_luns'):
                props.pop(key, None)
            yield props

    @staticmethod
    def _get_luns(con_props: dict, iqns=None) -> list:
        luns = con_props.get('target_luns')
        num_luns = len(con_props['target_iqns']) if iqns is None else len(iqns)
        return luns or [con_props['target_lun']] * num_luns

    def _get_all_targets(self, connection_properties: dict) -> list[tuple[str, str, list]]:
        if all((key in connection_properties for key in ('target_portals', 'target_iqns'))):
            return list(zip(connection_properties['target_portals'], connection_properties['target_iqns'], self._get_luns(connection_properties)))
        return [(connection_properties['target_portal'], connection_properties['target_iqn'], connection_properties.get('target_lun', 0))]