from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
class NuageEntityManager(object):
    """
    This module is meant to manage an entity in a Nuage VSP Platform
    """

    def __init__(self, module):
        self.module = module
        self.auth = module.params['auth']
        self.api_username = None
        self.api_password = None
        self.api_enterprise = None
        self.api_url = None
        self.api_version = None
        self.api_certificate = None
        self.api_key = None
        self.type = module.params['type']
        self.state = module.params['state']
        self.command = module.params['command']
        self.match_filter = module.params['match_filter']
        self.entity_id = module.params['id']
        self.parent_id = module.params['parent_id']
        self.parent_type = module.params['parent_type']
        self.properties = module.params['properties']
        self.children = module.params['children']
        self.entity = None
        self.entity_class = None
        self.parent = None
        self.parent_class = None
        self.entity_fetcher = None
        self.result = {'state': self.state, 'id': self.entity_id, 'entities': []}
        self.nuage_connection = None
        self._verify_api()
        self._verify_input()
        self._connect_vspk()
        self._find_parent()

    def _connect_vspk(self):
        """
        Connects to a Nuage API endpoint
        """
        try:
            if self.api_certificate and self.api_key:
                self.nuage_connection = VSPK.NUVSDSession(username=self.api_username, enterprise=self.api_enterprise, api_url=self.api_url, certificate=(self.api_certificate, self.api_key))
            else:
                self.nuage_connection = VSPK.NUVSDSession(username=self.api_username, password=self.api_password, enterprise=self.api_enterprise, api_url=self.api_url)
            self.nuage_connection.start()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to connect to the API URL with given username, password and enterprise: {0}'.format(error))

    def _verify_api(self):
        """
        Verifies the API and loads the proper VSPK version
        """
        if ('api_password' not in list(self.auth.keys()) or not self.auth['api_password']) and ('api_certificate' not in list(self.auth.keys()) or 'api_key' not in list(self.auth.keys()) or (not self.auth['api_certificate']) or (not self.auth['api_key'])):
            self.module.fail_json(msg='Missing api_password or api_certificate and api_key parameter in auth')
        self.api_username = self.auth['api_username']
        if 'api_password' in list(self.auth.keys()) and self.auth['api_password']:
            self.api_password = self.auth['api_password']
        if 'api_certificate' in list(self.auth.keys()) and 'api_key' in list(self.auth.keys()) and self.auth['api_certificate'] and self.auth['api_key']:
            self.api_certificate = self.auth['api_certificate']
            self.api_key = self.auth['api_key']
        self.api_enterprise = self.auth['api_enterprise']
        self.api_url = self.auth['api_url']
        self.api_version = self.auth['api_version']
        try:
            global VSPK
            VSPK = importlib.import_module('vspk.{0:s}'.format(self.api_version))
        except ImportError:
            self.module.fail_json(msg='vspk is required for this module, or the API version specified does not exist.')

    def _verify_input(self):
        """
        Verifies the parameter input for types and parent correctness and necessary parameters
        """
        try:
            self.entity_class = getattr(VSPK, 'NU{0:s}'.format(self.type))
        except AttributeError:
            self.module.fail_json(msg='Unrecognised type specified')
        if self.module.check_mode:
            return
        if self.parent_type:
            try:
                self.parent_class = getattr(VSPK, 'NU{0:s}'.format(self.parent_type))
            except AttributeError:
                self.module.fail_json(msg='Unrecognised parent type specified')
            fetcher = self.parent_class().fetcher_for_rest_name(self.entity_class.rest_name)
            if fetcher is None:
                self.module.fail_json(msg='Specified parent is not a valid parent for the specified type')
        elif not self.entity_id:
            self.parent_class = VSPK.NUMe
            fetcher = self.parent_class().fetcher_for_rest_name(self.entity_class.rest_name)
            if fetcher is None:
                self.module.fail_json(msg='No parent specified and root object is not a parent for the type')
        if self.command and self.command == 'change_password' and ('password' not in self.properties.keys()):
            self.module.fail_json(msg='command is change_password but the following are missing: password property')

    def _find_parent(self):
        """
        Fetches the parent if needed, otherwise configures the root object as parent. Also configures the entity fetcher
        Important notes:
        - If the parent is not set, the parent is automatically set to the root object
        - It the root object does not hold a fetcher for the entity, you have to provide an ID
        - If you want to assign/unassign, you have to provide a valid parent
        """
        self.parent = self.nuage_connection.user
        if self.parent_id:
            self.parent = self.parent_class(id=self.parent_id)
            try:
                self.parent.fetch()
            except BambouHTTPError as error:
                self.module.fail_json(msg='Failed to fetch the specified parent: {0}'.format(error))
        self.entity_fetcher = self.parent.fetcher_for_rest_name(self.entity_class.rest_name)

    def _find_entities(self, entity_id=None, entity_class=None, match_filter=None, properties=None, entity_fetcher=None):
        """
        Will return a set of entities matching a filter or set of properties if the match_filter is unset. If the
        entity_id is set, it will return only the entity matching that ID as the single element of the list.
        :param entity_id: Optional ID of the entity which should be returned
        :param entity_class: Optional class of the entity which needs to be found
        :param match_filter: Optional search filter
        :param properties: Optional set of properties the entities should contain
        :param entity_fetcher: The fetcher for the entity type
        :return: List of matching entities
        """
        search_filter = ''
        if entity_id:
            found_entity = entity_class(id=entity_id)
            try:
                found_entity.fetch()
            except BambouHTTPError as error:
                self.module.fail_json(msg='Failed to fetch the specified entity by ID: {0}'.format(error))
            return [found_entity]
        elif match_filter:
            search_filter = match_filter
        elif properties:
            for num, property_name in enumerate(properties):
                if num > 0:
                    search_filter += ' and '
                search_filter += '{0:s} == "{1}"'.format(property_name, properties[property_name])
        if entity_fetcher is not None:
            try:
                return entity_fetcher.get(filter=search_filter)
            except BambouHTTPError:
                pass
        return []

    def _find_entity(self, entity_id=None, entity_class=None, match_filter=None, properties=None, entity_fetcher=None):
        """
        Finds a single matching entity that matches all the provided properties, unless an ID is specified, in which
        case it just fetches the one item
        :param entity_id: Optional ID of the entity which should be returned
        :param entity_class: Optional class of the entity which needs to be found
        :param match_filter: Optional search filter
        :param properties: Optional set of properties the entities should contain
        :param entity_fetcher: The fetcher for the entity type
        :return: The first entity matching the criteria, or None if none was found
        """
        search_filter = ''
        if entity_id:
            found_entity = entity_class(id=entity_id)
            try:
                found_entity.fetch()
            except BambouHTTPError as error:
                self.module.fail_json(msg='Failed to fetch the specified entity by ID: {0}'.format(error))
            return found_entity
        elif match_filter:
            search_filter = match_filter
        elif properties:
            for num, property_name in enumerate(properties):
                if num > 0:
                    search_filter += ' and '
                search_filter += '{0:s} == "{1}"'.format(property_name, properties[property_name])
        if entity_fetcher is not None:
            try:
                return entity_fetcher.get_first(filter=search_filter)
            except BambouHTTPError:
                pass
        return None

    def handle_main_entity(self):
        """
        Handles the Ansible task
        """
        if self.command and self.command == 'find':
            self._handle_find()
        elif self.command and self.command == 'change_password':
            self._handle_change_password()
        elif self.command and self.command == 'wait_for_job':
            self._handle_wait_for_job()
        elif self.command and self.command == 'get_csp_enterprise':
            self._handle_get_csp_enterprise()
        elif self.state == 'present':
            self._handle_present()
        elif self.state == 'absent':
            self._handle_absent()
        self.module.exit_json(**self.result)

    def _handle_absent(self):
        """
        Handles the Ansible task when the state is set to absent
        """
        self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
        if self.entity and (self.entity_fetcher is None or self.entity_fetcher.relationship in ['child', 'root']):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                self._delete_entity(self.entity)
                self.result['id'] = None
        elif self.entity and self.entity_fetcher.relationship == 'member':
            if self._is_member(entity_fetcher=self.entity_fetcher, entity=self.entity):
                if self.module.check_mode:
                    self.result['changed'] = True
                else:
                    self._unassign_member(entity_fetcher=self.entity_fetcher, entity=self.entity, entity_class=self.entity_class, parent=self.parent, set_output=True)

    def _handle_present(self):
        """
        Handles the Ansible task when the state is set to present
        """
        self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
        if self.entity_fetcher is not None and self.entity_fetcher.relationship == 'member' and (not self.entity):
            self.module.fail_json(msg='Trying to assign an entity that does not exist')
        elif self.entity_fetcher is not None and self.entity_fetcher.relationship == 'member' and self.entity:
            if not self._is_member(entity_fetcher=self.entity_fetcher, entity=self.entity):
                if self.module.check_mode:
                    self.result['changed'] = True
                else:
                    self._assign_member(entity_fetcher=self.entity_fetcher, entity=self.entity, entity_class=self.entity_class, parent=self.parent, set_output=True)
        elif self.entity_fetcher is not None and self.entity_fetcher.relationship in ['child', 'root'] and (not self.entity):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                self.entity = self._create_entity(entity_class=self.entity_class, parent=self.parent, properties=self.properties)
                self.result['id'] = self.entity.id
                self.result['entities'].append(self.entity.to_dict())
            if self.children:
                for child in self.children:
                    self._handle_child(child=child, parent=self.entity)
        elif self.entity:
            changed = self._has_changed(entity=self.entity, properties=self.properties)
            if self.module.check_mode:
                self.result['changed'] = changed
            elif changed:
                self.entity = self._save_entity(entity=self.entity)
                self.result['id'] = self.entity.id
                self.result['entities'].append(self.entity.to_dict())
            else:
                self.result['id'] = self.entity.id
                self.result['entities'].append(self.entity.to_dict())
            if self.children:
                for child in self.children:
                    self._handle_child(child=child, parent=self.entity)
        elif not self.module.check_mode:
            self.module.fail_json(msg='Invalid situation, verify parameters')

    def _handle_get_csp_enterprise(self):
        """
        Handles the Ansible task when the command is to get the csp enterprise
        """
        self.entity_id = self.parent.enterprise_id
        self.entity = VSPK.NUEnterprise(id=self.entity_id)
        try:
            self.entity.fetch()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to fetch CSP enterprise: {0}'.format(error))
        self.result['id'] = self.entity_id
        self.result['entities'].append(self.entity.to_dict())

    def _handle_wait_for_job(self):
        """
        Handles the Ansible task when the command is to wait for a job
        """
        self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
        if self.module.check_mode:
            self.result['changed'] = True
        else:
            self._wait_for_job(self.entity)

    def _handle_change_password(self):
        """
        Handles the Ansible task when the command is to change a password
        """
        self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
        if self.module.check_mode:
            self.result['changed'] = True
        else:
            try:
                getattr(self.entity, 'password')
            except AttributeError:
                self.module.fail_json(msg='Entity does not have a password property')
            try:
                setattr(self.entity, 'password', self.properties['password'])
            except AttributeError:
                self.module.fail_json(msg='Password can not be changed for entity')
            self.entity = self._save_entity(entity=self.entity)
            self.result['id'] = self.entity.id
            self.result['entities'].append(self.entity.to_dict())

    def _handle_find(self):
        """
        Handles the Ansible task when the command is to find an entity
        """
        entities = self._find_entities(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
        self.result['changed'] = False
        if entities:
            if len(entities) == 1:
                self.result['id'] = entities[0].id
            for entity in entities:
                self.result['entities'].append(entity.to_dict())
        elif not self.module.check_mode:
            self.module.fail_json(msg='Unable to find matching entries')

    def _handle_child(self, child, parent):
        """
        Handles children of a main entity. Fields are similar to the normal fields
        Currently only supported state: present
        """
        if 'type' not in list(child.keys()):
            self.module.fail_json(msg='Child type unspecified')
        elif 'id' not in list(child.keys()) and 'properties' not in list(child.keys()):
            self.module.fail_json(msg='Child ID or properties unspecified')
        child_id = None
        if 'id' in list(child.keys()):
            child_id = child['id']
        child_properties = None
        if 'properties' in list(child.keys()):
            child_properties = child['properties']
        child_filter = None
        if 'match_filter' in list(child.keys()):
            child_filter = child['match_filter']
        entity_class = None
        try:
            entity_class = getattr(VSPK, 'NU{0:s}'.format(child['type']))
        except AttributeError:
            self.module.fail_json(msg='Unrecognised child type specified')
        entity_fetcher = parent.fetcher_for_rest_name(entity_class.rest_name)
        if entity_fetcher is None and (not child_id) and (not self.module.check_mode):
            self.module.fail_json(msg='Unable to find a fetcher for child, and no ID specified.')
        entity = self._find_entity(entity_id=child_id, entity_class=entity_class, match_filter=child_filter, properties=child_properties, entity_fetcher=entity_fetcher)
        if entity_fetcher.relationship == 'member' and (not entity):
            self.module.fail_json(msg='Trying to assign a child that does not exist')
        elif entity_fetcher.relationship == 'member' and entity:
            if not self._is_member(entity_fetcher=entity_fetcher, entity=entity):
                if self.module.check_mode:
                    self.result['changed'] = True
                else:
                    self._assign_member(entity_fetcher=entity_fetcher, entity=entity, entity_class=entity_class, parent=parent, set_output=False)
        elif entity_fetcher.relationship in ['child', 'root'] and (not entity):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                entity = self._create_entity(entity_class=entity_class, parent=parent, properties=child_properties)
        elif entity_fetcher.relationship in ['child', 'root'] and entity:
            changed = self._has_changed(entity=entity, properties=child_properties)
            if self.module.check_mode:
                self.result['changed'] = changed
            elif changed:
                entity = self._save_entity(entity=entity)
        if entity:
            self.result['entities'].append(entity.to_dict())
        if 'children' in list(child.keys()) and (not self.module.check_mode):
            for subchild in child['children']:
                self._handle_child(child=subchild, parent=entity)

    def _has_changed(self, entity, properties):
        """
        Compares a set of properties with a given entity, returns True in case the properties are different from the
        values in the entity
        :param entity: The entity to check
        :param properties: The properties to check
        :return: boolean
        """
        changed = False
        if properties:
            for property_name in list(properties.keys()):
                if property_name == 'password':
                    continue
                entity_value = ''
                try:
                    entity_value = getattr(entity, property_name)
                except AttributeError:
                    self.module.fail_json(msg='Property {0:s} is not valid for this type of entity'.format(property_name))
                if entity_value != properties[property_name]:
                    changed = True
                    try:
                        setattr(entity, property_name, properties[property_name])
                    except AttributeError:
                        self.module.fail_json(msg='Property {0:s} can not be changed for this type of entity'.format(property_name))
        return changed

    def _is_member(self, entity_fetcher, entity):
        """
        Verifies if the entity is a member of the parent in the fetcher
        :param entity_fetcher: The fetcher for the entity type
        :param entity: The entity to look for as a member in the entity fetcher
        :return: boolean
        """
        members = entity_fetcher.get()
        for member in members:
            if member.id == entity.id:
                return True
        return False

    def _assign_member(self, entity_fetcher, entity, entity_class, parent, set_output):
        """
        Adds the entity as a member to a parent
        :param entity_fetcher: The fetcher of the entity type
        :param entity: The entity to add as a member
        :param entity_class: The class of the entity
        :param parent: The parent on which to add the entity as a member
        :param set_output: If set to True, sets the Ansible result variables
        """
        members = entity_fetcher.get()
        members.append(entity)
        try:
            parent.assign(members, entity_class)
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to assign entity as a member: {0}'.format(error))
        self.result['changed'] = True
        if set_output:
            self.result['id'] = entity.id
            self.result['entities'].append(entity.to_dict())

    def _unassign_member(self, entity_fetcher, entity, entity_class, parent, set_output):
        """
        Removes the entity as a member of a parent
        :param entity_fetcher: The fetcher of the entity type
        :param entity: The entity to remove as a member
        :param entity_class: The class of the entity
        :param parent: The parent on which to add the entity as a member
        :param set_output: If set to True, sets the Ansible result variables
        """
        members = []
        for member in entity_fetcher.get():
            if member.id != entity.id:
                members.append(member)
        try:
            parent.assign(members, entity_class)
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to remove entity as a member: {0}'.format(error))
        self.result['changed'] = True
        if set_output:
            self.result['id'] = entity.id
            self.result['entities'].append(entity.to_dict())

    def _create_entity(self, entity_class, parent, properties):
        """
        Creates a new entity in the parent, with all properties configured as in the file
        :param entity_class: The class of the entity
        :param parent: The parent of the entity
        :param properties: The set of properties of the entity
        :return: The entity
        """
        entity = entity_class(**properties)
        try:
            parent.create_child(entity)
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to create entity: {0}'.format(error))
        self.result['changed'] = True
        return entity

    def _save_entity(self, entity):
        """
        Updates an existing entity
        :param entity: The entity to save
        :return: The updated entity
        """
        try:
            entity.save()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to update entity: {0}'.format(error))
        self.result['changed'] = True
        return entity

    def _delete_entity(self, entity):
        """
        Deletes an entity
        :param entity: The entity to delete
        """
        try:
            entity.delete()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Unable to delete entity: {0}'.format(error))
        self.result['changed'] = True

    def _wait_for_job(self, entity):
        """
        Waits for a job to finish
        :param entity: The job to wait for
        """
        running = False
        if entity.status == 'RUNNING':
            self.result['changed'] = True
            running = True
        while running:
            time.sleep(1)
            entity.fetch()
            if entity.status != 'RUNNING':
                running = False
        self.result['entities'].append(entity.to_dict())
        if entity.status == 'ERROR':
            self.module.fail_json(msg='Job ended in an error')