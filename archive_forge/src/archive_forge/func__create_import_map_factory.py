from os_ken.services.protocols.bgp.info_base.vrf import VrfRtImportMap
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4NlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6NlriImportMap
def _create_import_map_factory(self, name, value, cls):
    if self._import_maps_by_name.get(name) is not None:
        raise ImportMapAlreadyExistsError()
    self._import_maps_by_name[name] = cls(value)