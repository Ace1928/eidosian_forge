import logging
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.core_managers.import_map_manager\
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
def _create_rt_match_importmap(name, value):
    core_service = CORE_MANAGER.get_core_service()
    importmap_manager = core_service.importmap_manager
    try:
        importmap_manager.create_rt_import_map(name, value)
    except ImportMapAlreadyExistsError:
        raise RuntimeConfigError('Map with this name already exists')
    return True