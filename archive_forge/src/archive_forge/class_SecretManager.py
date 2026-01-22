import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
class SecretManager(base.BaseEntityManager):
    """Entity Manager for Secret entities"""

    def __init__(self, api):
        super(SecretManager, self).__init__(api, 'secrets')

    def get(self, secret_ref, payload_content_type=None):
        """Retrieve an existing Secret from Barbican

        :param str secret_ref: Full HATEOAS reference to a Secret, or a UUID
        :param str payload_content_type: DEPRECATED: Content type to use for
            payload decryption. Setting this can lead to unexpected results.
            See Launchpad Bug #1419166.
        :returns: Secret object retrieved from Barbican
        :rtype: :class:`barbicanclient.v1.secrets.Secret`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Getting secret - Secret href: {0}'.format(secret_ref))
        base.validate_ref_and_return_uuid(secret_ref, 'Secret')
        return Secret(api=self._api, payload_content_type=payload_content_type, secret_ref=secret_ref)

    def update(self, secret_ref, payload=None):
        """Update an existing Secret in Barbican

        :param str secret_ref: Full HATEOAS reference to a Secret, or a UUID
        :param str payload: New payload to add to secret
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        base.validate_ref_and_return_uuid(secret_ref, 'Secret')
        if not secret_ref:
            raise ValueError('secret_ref is required.')
        if type(payload) is bytes:
            headers = {'content-type': 'application/octet-stream'}
        elif type(payload) is str:
            headers = {'content-type': 'text/plain'}
        else:
            raise exceptions.PayloadException('Invalid Payload Type')
        uuid_ref = base.calculate_uuid_ref(secret_ref, self._entity)
        self._api.put(uuid_ref, headers=headers, data=payload)

    def create(self, name=None, payload=None, payload_content_type=None, payload_content_encoding=None, algorithm=None, bit_length=None, secret_type=None, mode=None, expiration=None):
        """Factory method for creating new `Secret` objects

        Secrets returned by this method have not yet been stored in the
        Barbican service.

        :param name: A friendly name for the Secret
        :param payload: The unencrypted secret data
        :param payload_content_type: DEPRECATED: The format/type of the secret
            data. Setting this can lead to unexpected results.  See Launchpad
            Bug #1419166.
        :param payload_content_encoding: DEPRECATED: The encoding of the secret
            data. Setting this can lead to unexpected results.  See Launchpad
            Bug #1419166.
        :param algorithm: The algorithm associated with this secret key
        :param bit_length: The bit length of this secret key
        :param mode: The algorithm mode used with this secret key
        :param secret_type: The secret type for this secret key
        :param expiration: The expiration time of the secret in ISO 8601 format
        :returns: A new Secret object
        :rtype: :class:`barbicanclient.v1.secrets.Secret`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        return Secret(api=self._api, name=name, payload=payload, payload_content_type=payload_content_type, payload_content_encoding=payload_content_encoding, algorithm=algorithm, bit_length=bit_length, mode=mode, secret_type=secret_type, expiration=expiration)

    def delete(self, secret_ref, force=False):
        """Delete a Secret from Barbican

        :param secret_ref: Full HATEOAS reference to a Secret, or a UUID
        :param force: When true, forces the deletion of secrets with consumers
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        base.validate_ref_and_return_uuid(secret_ref, 'Secret')
        if not secret_ref:
            raise ValueError('secret_ref is required.')
        secret_object = self.get(secret_ref=secret_ref)
        uuid_ref = base.calculate_uuid_ref(secret_ref, self._entity)
        if not secret_object.consumers or force:
            self._api.delete(uuid_ref)
        else:
            raise ValueError('Secret has consumers! Remove them first or use the force parameter to delete it.')

    def list(self, limit=10, offset=0, name=None, algorithm=None, mode=None, bits=0, secret_type=None, created=None, updated=None, expiration=None, sort=None):
        """List Secrets for the project

        This method uses the limit and offset parameters for paging,
        and also supports filtering.

        The time filters (created, updated, and expiration) are expected to
        be an ISO 8601 formatted string, which can be prefixed with comparison
        operators: 'gt:' (greater-than), 'gte:' (greater-than-or-equal), 'lt:'
        (less-than), or 'lte': (less-than-or-equal).

        :param limit: Max number of secrets returned
        :param offset: Offset secrets to begin list
        :param name: Name filter for the list
        :param algorithm: Algorithm filter for the list
        :param mode: Mode filter for the list
        :param bits: Bits filter for the list
        :param secret_type: Secret type filter for the list
        :param created: Created time filter for the list, an ISO 8601 format
            string, optionally prefixed with 'gt:', 'gte:', 'lt:', or 'lte:'
        :param updated: Updated time filter for the list, an ISO 8601 format
            string, optionally prefixed with 'gt:', 'gte:', 'lt:', or 'lte:'
        :param expiration: Expiration time filter for the list, an ISO 8601
            format string, optionally prefixed with 'gt:', 'gte:', 'lt:',
            or 'lte:'
        :param sort: Determines the sorted order of the returned list, a
            string of comma-separated sort keys ('created', 'expiration',
            'mode', 'name', 'secret_type', 'status', or 'updated') with a
            direction appended (':asc' or ':desc') to each key
        :returns: list of Secret objects that satisfy the provided filter
            criteria.
        :rtype: list
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Listing secrets - offset {0} limit {1}'.format(offset, limit))
        params = {'limit': limit, 'offset': offset}
        if name:
            params['name'] = name
        if algorithm:
            params['alg'] = algorithm
        if mode:
            params['mode'] = mode
        if bits > 0:
            params['bits'] = bits
        if secret_type:
            params['secret_type'] = secret_type
        if created:
            params['created'] = created
        if updated:
            params['updated'] = updated
        if expiration:
            params['expiration'] = expiration
        if sort:
            params['sort'] = sort
        response = self._api.get(self._entity, params=params)
        return [Secret(api=self._api, **s) for s in response.get('secrets', [])]

    def _enforce_microversion(self):
        if self._api.microversion == '1.0':
            raise NotImplementedError('Server does not support secret consumers.  Minimum key-manager microversion required: 1.1')

    def register_consumer(self, secret_ref, service, resource_type, resource_id):
        """Add a consumer to the secret

        :param secret_ref: Full HATEOAS reference to a secret, or a UUID
        :param service: Name of the consuming service
        :param resource_type: Type of the consuming resource
        :param resource_id: ID of the consuming resource
        :returns: A secret object per the get() method
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        :raises NotImplementedError: When using microversion 1.0
        """
        LOG.debug('Creating consumer registration for secret {0} of service {1} for resource type {2}with resource id {3}'.format(secret_ref, service, resource_type, resource_id))
        self._enforce_microversion()
        secret_uuid = base.validate_ref_and_return_uuid(secret_ref, 'Secret')
        href = '{0}/{1}/consumers'.format(self._entity, secret_uuid)
        consumer_dict = dict()
        consumer_dict['service'] = service
        consumer_dict['resource_type'] = resource_type
        consumer_dict['resource_id'] = resource_id
        response = self._api.post(href, json=consumer_dict)
        return Secret(api=self._api, **response)

    def remove_consumer(self, secret_ref, service, resource_type, resource_id):
        """Remove a consumer from the secret

        :param secret_ref: Full HATEOAS reference to a secret, or a UUID
        :param service: Name of the previously consuming service
        :param resource_type: type of the previously consuming resource
        :param resource_id: ID of the previously consuming resource
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Deleting consumer registration for secret {0} of service {1} for resource type {2}with resource id {3}'.format(secret_ref, service, resource_type, resource_id))
        self._enforce_microversion()
        secret_uuid = base.validate_ref_and_return_uuid(secret_ref, 'secret')
        href = '{0}/{1}/consumers'.format(self._entity, secret_uuid)
        consumer_dict = {'service': service, 'resource_type': resource_type, 'resource_id': resource_id}
        self._api.delete(href, json=consumer_dict)

    def list_consumers(self, secret_ref, limit=10, offset=0):
        """List consumers of the secret

        :param secret_ref: Full HATEOAS reference to a secret, or a UUID
        :param limit: Max number of consumers returned
        :param offset: Offset secrets to begin list
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Listing consumers of secret {0}'.format(secret_ref))
        self._enforce_microversion()
        secret_uuid = base.validate_ref_and_return_uuid(secret_ref, 'secret')
        href = '{0}/{1}/consumers'.format(self._entity, secret_uuid)
        params = {'limit': limit, 'offset': offset}
        response = self._api.get(href, params=params)
        return [SecretConsumers(secret_ref=secret_ref, **s) for s in response.get('consumers', [])]