from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def __replace_name_by_uris(self, data):
    map_template = data.get('interconnectMapTemplate')
    if map_template:
        map_entry_templates = map_template.get('interconnectMapEntryTemplates')
        if map_entry_templates:
            for value in map_entry_templates:
                permitted_interconnect_type_name = value.pop('permittedInterconnectTypeName', None)
                if permitted_interconnect_type_name:
                    value['permittedInterconnectTypeUri'] = self.__get_interconnect_type_by_name(permitted_interconnect_type_name).get('uri')