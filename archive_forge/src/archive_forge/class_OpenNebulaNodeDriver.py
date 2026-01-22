import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebulaNodeDriver(NodeDriver):
    """
    OpenNebula.org node driver.
    """
    connectionCls = OpenNebulaConnection
    name = 'OpenNebula'
    website = 'http://opennebula.org/'
    type = Provider.OPENNEBULA
    NODE_STATE_MAP = {'INIT': NodeState.PENDING, 'PENDING': NodeState.PENDING, 'HOLD': NodeState.PENDING, 'ACTIVE': NodeState.RUNNING, 'STOPPED': NodeState.TERMINATED, 'SUSPENDED': NodeState.PENDING, 'DONE': NodeState.TERMINATED, 'FAILED': NodeState.TERMINATED}

    def __new__(cls, key, secret=None, api_version=DEFAULT_API_VERSION, **kwargs):
        if cls is OpenNebulaNodeDriver:
            if api_version in ['1.4']:
                cls = OpenNebula_1_4_NodeDriver
            elif api_version in ['2.0', '2.2']:
                cls = OpenNebula_2_0_NodeDriver
            elif api_version in ['3.0']:
                cls = OpenNebula_3_0_NodeDriver
            elif api_version in ['3.2']:
                cls = OpenNebula_3_2_NodeDriver
            elif api_version in ['3.6']:
                cls = OpenNebula_3_6_NodeDriver
            elif api_version in ['3.8']:
                cls = OpenNebula_3_8_NodeDriver
                if 'plain_auth' not in kwargs:
                    kwargs['plain_auth'] = cls.plain_auth
                else:
                    cls.plain_auth = kwargs['plain_auth']
            else:
                raise NotImplementedError('No OpenNebulaNodeDriver found for API version %s' % api_version)
            return super().__new__(cls)

    def create_node(self, name, size, image, networks=None):
        """
        Create a new OpenNebula node.

        @inherits: :class:`NodeDriver.create_node`

        :keyword networks: List of virtual networks to which this node should
                           connect. (optional)
        :type    networks: :class:`OpenNebulaNetwork` or
            ``list`` of :class:`OpenNebulaNetwork`
        """
        compute = ET.Element('COMPUTE')
        name = ET.SubElement(compute, 'NAME')
        name.text = name
        instance_type = ET.SubElement(compute, 'INSTANCE_TYPE')
        instance_type.text = size.name
        storage = ET.SubElement(compute, 'STORAGE')
        ET.SubElement(storage, 'DISK', {'image': '%s' % str(image.id)})
        if networks:
            if not isinstance(networks, list):
                networks = [networks]
            networkGroup = ET.SubElement(compute, 'NETWORK')
            for network in networks:
                if network.address:
                    ET.SubElement(networkGroup, 'NIC', {'network': '%s' % str(network.id), 'ip': network.address})
                else:
                    ET.SubElement(networkGroup, 'NIC', {'network': '%s' % str(network.id)})
        xml = ET.tostring(compute)
        node = self.connection.request('/compute', method='POST', data=xml).object
        return self._to_node(node)

    def destroy_node(self, node):
        url = '/compute/%s' % str(node.id)
        resp = self.connection.request(url, method='DELETE')
        return resp.status == httplib.OK

    def list_nodes(self):
        return self._to_nodes(self.connection.request('/compute').object)

    def list_images(self, location=None):
        return self._to_images(self.connection.request('/storage').object)

    def list_sizes(self, location=None):
        """
        Return list of sizes on a provider.

        @inherits: :class:`NodeDriver.list_sizes`

        :return: List of compute node sizes supported by the cloud provider.
        :rtype:  ``list`` of :class:`OpenNebulaNodeSize`
        """
        return [NodeSize(id=1, name='small', ram=None, disk=None, bandwidth=None, price=None, driver=self), NodeSize(id=2, name='medium', ram=None, disk=None, bandwidth=None, price=None, driver=self), NodeSize(id=3, name='large', ram=None, disk=None, bandwidth=None, price=None, driver=self)]

    def list_locations(self):
        return [NodeLocation(0, '', '', self)]

    def ex_list_networks(self, location=None):
        """
        List virtual networks on a provider.

        :param location: Location from which to request a list of virtual
                         networks. (optional)
        :type  location: :class:`NodeLocation`

        :return: List of virtual networks available to be connected to a
                 compute node.
        :rtype:  ``list`` of :class:`OpenNebulaNetwork`
        """
        return self._to_networks(self.connection.request('/network').object)

    def ex_node_action(self, node, action):
        """
        Build action representation and instruct node to commit action.

        Build action representation from the compute node ID, and the
        action which should be carried out on that compute node. Then
        instruct the node to carry out that action.

        :param node: Compute node instance.
        :type  node: :class:`Node`

        :param action: Action to be carried out on the compute node.
        :type  action: ``str``

        :return: False if an HTTP Bad Request is received, else, True is
                 returned.
        :rtype:  ``bool``
        """
        compute_node_id = str(node.id)
        compute = ET.Element('COMPUTE')
        compute_id = ET.SubElement(compute, 'ID')
        compute_id.text = compute_node_id
        state = ET.SubElement(compute, 'STATE')
        state.text = action
        xml = ET.tostring(compute)
        url = '/compute/%s' % compute_node_id
        resp = self.connection.request(url, method='PUT', data=xml)
        if resp.status == httplib.BAD_REQUEST:
            return False
        else:
            return True

    def _to_images(self, object):
        """
        Request a list of images and convert that list to a list of NodeImage
        objects.

        Request a list of images from the OpenNebula web interface, and
        issue a request to convert each XML object representation of an image
        to a NodeImage object.

        :rtype:  ``list`` of :class:`NodeImage`
        :return: List of images.
        """
        images = []
        for element in object.findall('DISK'):
            image_id = element.attrib['href'].partition('/storage/')[2]
            image = self.connection.request('/storage/%s' % image_id).object
            images.append(self._to_image(image))
        return images

    def _to_image(self, image):
        """
        Take XML object containing an image description and convert to
        NodeImage object.

        :type  image: :class:`ElementTree`
        :param image: XML representation of an image.

        :rtype:  :class:`NodeImage`
        :return: The newly extracted :class:`NodeImage`.
        """
        return NodeImage(id=image.findtext('ID'), name=image.findtext('NAME'), driver=self.connection.driver, extra={'size': image.findtext('SIZE'), 'url': image.findtext('URL')})

    def _to_networks(self, object):
        """
        Request a list of networks and convert that list to a list of
        OpenNebulaNetwork objects.

        Request a list of networks from the OpenNebula web interface, and
        issue a request to convert each XML object representation of a network
        to an OpenNebulaNetwork object.

        :rtype:  ``list`` of :class:`OpenNebulaNetwork`
        :return: List of virtual networks.
        """
        networks = []
        for element in object.findall('NETWORK'):
            network_id = element.attrib['href'].partition('/network/')[2]
            network_element = self.connection.request('/network/%s' % network_id).object
            networks.append(self._to_network(network_element))
        return networks

    def _to_network(self, element):
        """
        Take XML object containing a network description and convert to
        OpenNebulaNetwork object.

        Take XML representation containing a network description and
        convert to OpenNebulaNetwork object.

        :rtype:  :class:`OpenNebulaNetwork`
        :return: The newly extracted :class:`OpenNebulaNetwork`.
        """
        return OpenNebulaNetwork(id=element.findtext('ID'), name=element.findtext('NAME'), address=element.findtext('ADDRESS'), size=element.findtext('SIZE'), driver=self.connection.driver)

    def _to_nodes(self, object):
        """
        Request a list of compute nodes and convert that list to a list of
        Node objects.

        Request a list of compute nodes from the OpenNebula web interface, and
        issue a request to convert each XML object representation of a node
        to a Node object.

        :rtype:  ``list`` of :class:`Node`
        :return: A list of compute nodes.
        """
        computes = []
        for element in object.findall('COMPUTE'):
            compute_id = element.attrib['href'].partition('/compute/')[2]
            compute = self.connection.request('/compute/%s' % compute_id).object
            computes.append(self._to_node(compute))
        return computes

    def _to_node(self, compute):
        """
        Take XML object containing a compute node description and convert to
        Node object.

        Take XML representation containing a compute node description and
        convert to Node object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  :class:`Node`
        :return: The newly extracted :class:`Node`.
        """
        try:
            state = self.NODE_STATE_MAP[compute.findtext('STATE').upper()]
        except KeyError:
            state = NodeState.UNKNOWN
        return Node(id=compute.findtext('ID'), name=compute.findtext('NAME'), state=state, public_ips=self._extract_networks(compute), private_ips=[], driver=self.connection.driver, image=self._extract_images(compute))

    def _extract_networks(self, compute):
        """
        Extract networks from a compute node XML representation.

        Extract network descriptions from a compute node XML representation,
        converting each network to an OpenNebulaNetwork object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``list`` of :class:`OpenNebulaNetwork`s.
        :return: List of virtual networks attached to the compute node.
        """
        networks = list()
        network_list = compute.find('NETWORK')
        for element in network_list.findall('NIC'):
            networks.append(OpenNebulaNetwork(id=element.attrib.get('network', None), name=None, address=element.attrib.get('ip', None), size=1, driver=self.connection.driver))
        return networks

    def _extract_images(self, compute):
        """
        Extract image disks from a compute node XML representation.

        Extract image disk descriptions from a compute node XML representation,
        converting the disks to an NodeImage object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  :class:`NodeImage`.
        :return: First disk attached to a compute node.
        """
        disks = list()
        disk_list = compute.find('STORAGE')
        if disk_list is not None:
            for element in disk_list.findall('DISK'):
                disks.append(NodeImage(id=element.attrib.get('image', None), name=None, driver=self.connection.driver, extra={'dev': element.attrib.get('dev', None)}))
        if len(disks) > 0:
            return disks[0]
        else:
            return None