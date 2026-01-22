import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
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