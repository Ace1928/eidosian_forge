import logging
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.core_managers.import_map_manager\
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
def _create_prefix_match_importmap(name, value, route_family):
    core_service = CORE_MANAGER.get_core_service()
    importmap_manager = core_service.importmap_manager
    try:
        if route_family == 'ipv4':
            importmap_manager.create_vpnv4_nlri_import_map(name, value)
        elif route_family == 'ipv6':
            importmap_manager.create_vpnv6_nlri_import_map(name, value)
        else:
            raise RuntimeConfigError('Unknown address family %s. it should be ipv4 or ipv6' % route_family)
    except ImportMapAlreadyExistsError:
        raise RuntimeConfigError('Map with this name already exists')
    return True