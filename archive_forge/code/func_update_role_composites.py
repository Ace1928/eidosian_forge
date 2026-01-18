from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_role_composites(self, rolerep, composites, clientid=None, realm='master'):
    existing_composites = self.get_role_composites(rolerep=rolerep, clientid=clientid, realm=realm)
    composites_to_be_created = []
    composites_to_be_deleted = []
    for composite in composites:
        composite_found = False
        existing_composite_client = None
        for existing_composite in existing_composites:
            if existing_composite['clientRole']:
                existing_composite_client = self.get_client_by_id(existing_composite['containerId'], realm=realm)
                if 'client_id' in composite and composite['client_id'] is not None and (existing_composite_client['clientId'] == composite['client_id']) and (composite['name'] == existing_composite['name']):
                    composite_found = True
                    break
            elif ('client_id' not in composite or composite['client_id'] is None) and composite['name'] == existing_composite['name']:
                composite_found = True
                break
        if not composite_found and ('state' not in composite or composite['state'] == 'present'):
            if 'client_id' in composite and composite['client_id'] is not None:
                client_roles = self.get_client_roles(clientid=composite['client_id'], realm=realm)
                for client_role in client_roles:
                    if client_role['name'] == composite['name']:
                        composites_to_be_created.append(client_role)
                        break
            else:
                realm_role = self.get_realm_role(name=composite['name'], realm=realm)
                composites_to_be_created.append(realm_role)
        elif composite_found and 'state' in composite and (composite['state'] == 'absent'):
            if 'client_id' in composite and composite['client_id'] is not None:
                client_roles = self.get_client_roles(clientid=composite['client_id'], realm=realm)
                for client_role in client_roles:
                    if client_role['name'] == composite['name']:
                        composites_to_be_deleted.append(client_role)
                        break
            else:
                realm_role = self.get_realm_role(name=composite['name'], realm=realm)
                composites_to_be_deleted.append(realm_role)
    if len(composites_to_be_created) > 0:
        self.create_role_composites(rolerep=rolerep, composites=composites_to_be_created, clientid=clientid, realm=realm)
    if len(composites_to_be_deleted) > 0:
        self.delete_role_composites(rolerep=rolerep, composites=composites_to_be_deleted, clientid=clientid, realm=realm)