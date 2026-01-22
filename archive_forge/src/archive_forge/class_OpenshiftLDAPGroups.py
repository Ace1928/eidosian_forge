from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
class OpenshiftLDAPGroups(object):
    kind = 'Group'
    version = 'user.openshift.io/v1'

    def __init__(self, module):
        self.module = module
        self.cache = {}
        self.__group_api = None

    @property
    def k8s_group_api(self):
        if not self.__group_api:
            params = dict(kind=self.kind, api_version=self.version, fail=True)
            self.__group_api = self.module.find_resource(**params)
        return self.__group_api

    def get_group_info(self, return_list=False, **kwargs):
        params = dict(kind=self.kind, api_version=self.version)
        params.update(kwargs)
        result = self.module.kubernetes_facts(**params)
        if len(result['resources']) == 0:
            return None
        if len(result['resources']) == 1 and (not return_list):
            return result['resources'][0]
        else:
            return result['resources']

    def list_groups(self):
        allow_groups = self.module.params.get('allow_groups')
        deny_groups = self.module.params.get('deny_groups')
        name_mapping = self.module.config.get('groupUIDNameMapping')
        if name_mapping and (allow_groups or deny_groups):

            def _map_group_names(groups):
                return [name_mapping.get(value, value) for value in groups]
            allow_groups = _map_group_names(allow_groups)
            deny_groups = _map_group_names(deny_groups)
        host = self.module.host
        netlocation = self.module.netlocation
        groups = []
        if allow_groups:
            missing = []
            for grp in allow_groups:
                if grp in deny_groups:
                    continue
                resource = self.get_group_info(name=grp)
                if not resource:
                    missing.append(grp)
                    continue
                groups.append(resource)
            if missing:
                self.module.fail_json(msg='The following groups were not found: %s' % ''.join(missing))
        else:
            label_selector = '%s=%s' % (LDAP_OPENSHIFT_HOST_LABEL, host)
            resources = self.get_group_info(label_selectors=[label_selector], return_list=True)
            if not resources:
                return (None, "Unable to find Group matching label selector '%s'" % label_selector)
            groups = resources
            if deny_groups:
                groups = [item for item in groups if item['metadata']['name'] not in deny_groups]
        uids = []
        for grp in groups:
            err = validate_group_annotation(grp, netlocation)
            if err and allow_groups:
                return (None, err)
            group_uid = grp['metadata']['annotations'].get(LDAP_OPENSHIFT_UID_ANNOTATION)
            self.cache[group_uid] = grp
            uids.append(group_uid)
        return (uids, None)

    def get_group_name_for_uid(self, group_uid):
        if group_uid not in self.cache:
            return (None, 'No mapping found for Group uid: %s' % group_uid)
        return (self.cache[group_uid]['metadata']['name'], None)

    def make_openshift_group(self, group_uid, group_name, usernames):
        group = self.get_group_info(name=group_name)
        if not group:
            group = {'apiVersion': 'user.openshift.io/v1', 'kind': 'Group', 'metadata': {'name': group_name, 'labels': {LDAP_OPENSHIFT_HOST_LABEL: self.module.host}, 'annotations': {LDAP_OPENSHIFT_URL_ANNOTATION: self.module.netlocation, LDAP_OPENSHIFT_UID_ANNOTATION: group_uid}}}
        ldaphost_label = group['metadata'].get('labels', {}).get(LDAP_OPENSHIFT_HOST_LABEL)
        if not ldaphost_label or ldaphost_label != self.module.host:
            return (None, 'Group %s: %s label did not match sync host: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_HOST_LABEL, self.module.host, ldaphost_label))
        ldapurl_annotation = group['metadata'].get('annotations', {}).get(LDAP_OPENSHIFT_URL_ANNOTATION)
        if not ldapurl_annotation or ldapurl_annotation != self.module.netlocation:
            return (None, 'Group %s: %s annotation did not match sync host: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_URL_ANNOTATION, self.module.netlocation, ldapurl_annotation))
        ldapuid_annotation = group['metadata'].get('annotations', {}).get(LDAP_OPENSHIFT_UID_ANNOTATION)
        if not ldapuid_annotation or ldapuid_annotation != group_uid:
            return (None, 'Group %s: %s annotation did not match LDAP UID: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_UID_ANNOTATION, group_uid, ldapuid_annotation))
        group['users'] = usernames
        group['metadata']['annotations'][LDAP_OPENSHIFT_SYNCTIME_ANNOTATION] = datetime.now().isoformat()
        return (group, None)

    def create_openshift_groups(self, groups: list):
        diffs = []
        results = []
        changed = False
        for definition in groups:
            name = definition['metadata']['name']
            existing = self.get_group_info(name=name)
            if not self.module.check_mode:
                method = 'patch' if existing else 'create'
                try:
                    if existing:
                        definition = self.k8s_group_api.patch(definition).to_dict()
                    else:
                        definition = self.k8s_group_api.create(definition).to_dict()
                except DynamicApiError as exc:
                    self.module.fail_json(msg="Failed to %s Group '%s' due to: %s" % (method, name, exc.body))
                except Exception as exc:
                    self.module.fail_json(msg="Failed to %s Group '%s' due to: %s" % (method, name, to_native(exc)))
            equals = False
            if existing:
                equals, diff = self.module.diff_objects(existing, definition)
                diffs.append(diff)
            changed = changed or not equals
            results.append(definition)
        return (results, diffs, changed)

    def delete_openshift_group(self, name: str):
        result = dict(kind=self.kind, apiVersion=self.version, metadata=dict(name=name))
        if not self.module.check_mode:
            try:
                result = self.k8s_group_api.delete(name=name).to_dict()
            except DynamicApiError as exc:
                self.module.fail_json(msg="Failed to delete Group '{0}' due to: {1}".format(name, exc.body))
            except Exception as exc:
                self.module.fail_json(msg="Failed to delete Group '{0}' due to: {1}".format(name, to_native(exc)))
        return result