import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class OrderManager(base.BaseEntityManager):
    """Entity Manager for Order entitites"""
    _order_type_to_class_map = {'key': KeyOrder, 'asymmetric': AsymmetricOrder, 'certificate': CertificateOrder}

    def __init__(self, api):
        super(OrderManager, self).__init__(api, 'orders')

    def get(self, order_ref):
        """Retrieve an existing Order from Barbican

        :param order_ref: Full HATEOAS reference to an Order, or a UUID
        :returns: An instance of the appropriate subtype of Order
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Getting order - Order href: {0}'.format(order_ref))
        uuid_ref = base.calculate_uuid_ref(order_ref, self._entity)
        try:
            response = self._api.get(uuid_ref)
        except AttributeError:
            raise LookupError('Order {0} could not be found.'.format(order_ref))
        return self._create_typed_order(response)

    def _create_typed_order(self, response):
        resp_type = response.pop('type').lower()
        order_type = self._order_type_to_class_map.get(resp_type)
        if resp_type == 'certificate' and 'container_ref' in response.get('meta', ()):
            response['source_container_ref'] = response['meta'].pop('container_ref')
        if resp_type == 'key' and set(response['meta'].keys()) - set(KeyOrder._validMeta):
            invalidFields = ', '.join(map(str, set(response['meta'].keys()) - set(KeyOrder._validMeta)))
            raise TypeError('Invalid KeyOrder meta field: [%s]' % invalidFields)
        response.update(response.pop('meta'))
        if order_type is not None:
            return order_type(self._api, **response)
        else:
            raise TypeError('Unknown Order type "{0}"'.format(order_type))

    def create(self, type=None, **kwargs):
        if not type:
            raise TypeError('No Order type provided')
        order_type = self._order_type_to_class_map.get(type.lower())
        if not order_type:
            raise TypeError('Unknown Order type "{0}"'.format(type))
        return order_type(self._api, **kwargs)

    def create_key(self, name=None, algorithm=None, bit_length=None, mode=None, payload_content_type=None, expiration=None):
        """Factory method for `KeyOrder` objects

        `KeyOrder` objects returned by this method have not yet been submitted
        to the Barbican service.

        :param name: A friendly name for the secret to be created
        :param algorithm: The algorithm associated with this secret key
        :param bit_length: The bit length of this secret key
        :param mode: The algorithm mode used with this secret key
        :param payload_content_type: The format/type of the secret data
        :param expiration: The expiration time of the secret in ISO 8601 format
        :returns: KeyOrder
        :rtype: :class:`barbicanclient.v1.orders.KeyOrder`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        return KeyOrder(api=self._api, name=name, algorithm=algorithm, bit_length=bit_length, mode=mode, payload_content_type=payload_content_type, expiration=expiration)

    def create_asymmetric(self, name=None, algorithm=None, bit_length=None, pass_phrase=None, payload_content_type=None, expiration=None):
        """Factory method for `AsymmetricOrder` objects

        `AsymmetricOrder` objects returned by this method have not yet been
        submitted to the Barbican service.

        :param name: A friendly name for the container to be created
        :param algorithm: The algorithm associated with this secret key
        :param bit_length: The bit length of this secret key
        :param pass_phrase: Optional passphrase
        :param payload_content_type: The format/type of the secret data
        :param expiration: The expiration time of the secret in ISO 8601 format
        :returns: AsymmetricOrder
        :rtype: :class:`barbicanclient.v1.orders.AsymmetricOrder`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        return AsymmetricOrder(api=self._api, name=name, algorithm=algorithm, bit_length=bit_length, passphrase=pass_phrase, payload_content_type=payload_content_type, expiration=expiration)

    def create_certificate(self, name=None, request_type=None, subject_dn=None, source_container_ref=None, ca_id=None, profile=None, request_data=None):
        """Factory method for `CertificateOrder` objects

        `CertificateOrder` objects returned by this method have not yet been
        submitted to the Barbican service.

        :param name: A friendly name for the container to be created
        :param request_type: The type of the certificate request
        :param subject_dn: A subject for the certificate
        :param source_container_ref: A container with a public/private key pair
            to use as source for stored-key requests
        :param ca_id: The identifier of the CA to use
        :param profile: The profile of certificate to use
        :param request_data: The CSR content
        :returns: CertificateOrder
        :rtype: :class:`barbicanclient.v1.orders.CertificateOrder`
        """
        return CertificateOrder(api=self._api, name=name, request_type=request_type, subject_dn=subject_dn, source_container_ref=source_container_ref, ca_id=ca_id, profile=profile, request_data=request_data)

    def delete(self, order_ref):
        """Delete an Order from Barbican

        :param order_ref: Full HATEOAS reference to an Order, or a UUID
        """
        if not order_ref:
            raise ValueError('order_ref is required.')
        uuid_ref = base.calculate_uuid_ref(order_ref, self._entity)
        self._api.delete(uuid_ref)

    def list(self, limit=10, offset=0):
        """List Orders for the project

        This method uses the limit and offset parameters for paging.

        :param limit: Max number of orders returned
        :param offset: Offset orders to begin list
        :returns: list of Order objects
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Listing orders - offset {0} limit {1}'.format(offset, limit))
        params = {'limit': limit, 'offset': offset}
        response = self._api.get(self._entity, params=params)
        return [self._create_typed_order(o) for o in response.get('orders', [])]