from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
class ManageIQ(object):
    """
        class encapsulating ManageIQ API client.
    """

    def __init__(self, module):
        check_client(module)
        params = validate_connection_params(module)
        url = params['url']
        username = params['username']
        password = params['password']
        token = params['token']
        verify_ssl = params['validate_certs']
        ca_bundle_path = params['ca_cert']
        self._module = module
        self._api_url = url + '/api'
        self._auth = dict(user=username, password=password, token=token)
        try:
            self._client = ManageIQClient(self._api_url, self._auth, verify_ssl=verify_ssl, ca_bundle_path=ca_bundle_path)
        except Exception as e:
            self.module.fail_json(msg='failed to open connection (%s): %s' % (url, str(e)))

    @property
    def module(self):
        """ Ansible module module

        Returns:
            the ansible module
        """
        return self._module

    @property
    def api_url(self):
        """ Base ManageIQ API

        Returns:
            the base ManageIQ API
        """
        return self._api_url

    @property
    def client(self):
        """ ManageIQ client

        Returns:
            the ManageIQ client
        """
        return self._client

    def find_collection_resource_by(self, collection_name, **params):
        """ Searches the collection resource by the collection name and the param passed.

        Returns:
            the resource as an object if it exists in manageiq, None otherwise.
        """
        try:
            entity = self.client.collections.__getattribute__(collection_name).get(**params)
        except ValueError:
            return None
        except Exception as e:
            self.module.fail_json(msg='failed to find resource {error}'.format(error=e))
        return vars(entity)

    def find_collection_resource_or_fail(self, collection_name, **params):
        """ Searches the collection resource by the collection name and the param passed.

        Returns:
            the resource as an object if it exists in manageiq, Fail otherwise.
        """
        resource = self.find_collection_resource_by(collection_name, **params)
        if resource:
            return resource
        else:
            msg = '{collection_name} where {params} does not exist in manageiq'.format(collection_name=collection_name, params=str(params))
            self.module.fail_json(msg=msg)

    def policies(self, resource_id, resource_type, resource_name):
        manageiq = ManageIQ(self.module)
        if resource_id is None:
            resource_id = manageiq.find_collection_resource_or_fail(resource_type, name=resource_name)['id']
        return ManageIQPolicies(manageiq, resource_type, resource_id)

    def query_resource_id(self, resource_type, resource_name):
        """ Query the resource name in ManageIQ.

        Returns:
            the resource ID if it exists in ManageIQ, Fail otherwise.
        """
        resource = self.find_collection_resource_by(resource_type, name=resource_name)
        if resource:
            return resource['id']
        else:
            msg = '{resource_name} {resource_type} does not exist in manageiq'.format(resource_name=resource_name, resource_type=resource_type)
            self.module.fail_json(msg=msg)