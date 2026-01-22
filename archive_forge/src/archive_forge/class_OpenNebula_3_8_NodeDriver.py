import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebula_3_8_NodeDriver(OpenNebula_3_6_NodeDriver):
    """
    OpenNebula.org node driver for OpenNebula.org v3.8.
    """
    name = 'OpenNebula (v3.8)'
    plain_auth = API_PLAIN_AUTH

    def _to_sizes(self, object):
        """
        Request a list of instance types and convert that list to a list of
        OpenNebulaNodeSize objects.

        Request a list of instance types from the OpenNebula web interface,
        and issue a request to convert each XML object representation of an
        instance type to an OpenNebulaNodeSize object.

        :return: List of instance types.
        :rtype:  ``list`` of :class:`OpenNebulaNodeSize`
        """
        sizes = []
        size_id = 1
        attributes = [('name', str, None), ('ram', int, 'MEMORY'), ('cpu', float, None), ('vcpu', float, None), ('disk', str, None), ('bandwidth', float, None), ('price', float, None)]
        for element in object.findall('INSTANCE_TYPE'):
            element = self.connection.request('/instance_type/%s' % element.attrib['name']).object
            size_kwargs = {'id': size_id, 'driver': self}
            values = self._get_attributes_values(attributes=attributes, element=element)
            size_kwargs.update(values)
            size = OpenNebulaNodeSize(**size_kwargs)
            sizes.append(size)
            size_id += 1
        return sizes

    def _ex_connection_class_kwargs(self):
        """
        Set plain_auth as an extra :class:`OpenNebulaConnection_3_8` argument

        :return: ``dict`` of :class:`OpenNebulaConnection_3_8` input arguments
        """
        return {'plain_auth': self.plain_auth}