from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
class ElementSWCluster(object):
    """
    Element Software Initialize node with ownership for cluster formation
    """

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(management_virtual_ip=dict(required=True, type='str'), storage_virtual_ip=dict(required=True, type='str'), replica_count=dict(required=False, type='int', default=2), cluster_admin_username=dict(required=False, type='str'), cluster_admin_password=dict(required=False, type='str', no_log=True), accept_eula=dict(required=False, type='bool'), nodes=dict(required=True, type='list', elements='str'), attributes=dict(required=False, type='dict', default=None), timeout=dict(required=False, type='int', default=100), fail_if_cluster_already_exists_with_larger_ensemble=dict(required=False, type='bool', default=True), encryption=dict(required=False, type='bool'), order_number=dict(required=False, type='str'), serial_number=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        input_params = self.module.params
        self.management_virtual_ip = input_params['management_virtual_ip']
        self.storage_virtual_ip = input_params['storage_virtual_ip']
        self.replica_count = input_params['replica_count']
        self.accept_eula = input_params.get('accept_eula')
        self.attributes = input_params.get('attributes')
        self.nodes = input_params['nodes']
        self.cluster_admin_username = input_params['username'] if input_params.get('cluster_admin_username') is None else input_params['cluster_admin_username']
        self.cluster_admin_password = input_params['password'] if input_params.get('cluster_admin_password') is None else input_params['cluster_admin_password']
        self.fail_if_cluster_already_exists_with_larger_ensemble = input_params['fail_if_cluster_already_exists_with_larger_ensemble']
        self.encryption = input_params['encryption']
        self.order_number = input_params['order_number']
        self.serial_number = input_params['serial_number']
        self.debug = list()
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        for role, port in [('node', 442), ('cluster', 443)]:
            try:
                conn = netapp_utils.create_sf_connection(module=self.module, raise_on_connection_error=True, port=port, timeout=input_params['timeout'])
                if role == 'node':
                    self.sfe_node = conn
                else:
                    self.sfe_cluster = conn
            except netapp_utils.solidfire.common.ApiConnectionError as exc:
                if str(exc) == 'Bad Credentials':
                    msg = 'Most likely the cluster is already created.'
                    msg += '  Make sure to use valid %s credentials for username and password.' % 'node' if port == 442 else 'cluster'
                    msg += '  Even though credentials are not required for the first create, they are needed to check whether the cluster already exists.'
                    msg += '  Cluster reported: %s' % repr(exc)
                else:
                    msg = 'Failed to create connection: %s' % repr(exc)
                self.module.fail_json(msg=msg)
            except Exception as exc:
                self.module.fail_json(msg='Failed to connect: %s' % repr(exc))
        self.elementsw_helper = NaElementSWModule(self.sfe_cluster)
        if self.attributes is not None:
            self.attributes.update(self.elementsw_helper.set_element_attributes(source='na_elementsw_cluster'))
        else:
            self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_cluster')

    def get_node_cluster_info(self):
        """
        Get Cluster Info - using node API
        """
        try:
            info = self.sfe_node.get_config()
            self.debug.append(repr(info.config.cluster))
            return info.config.cluster
        except Exception as exc:
            self.debug.append('port: %s, %s' % (str(self.sfe_node._port), repr(exc)))
            return None

    def check_cluster_exists(self):
        """
        validate if cluster exists with list of nodes
        error out if something is found but with different nodes
        return a tuple (found, info)
            found is True if found, False if not found
        """
        info = self.get_node_cluster_info()
        if info is None:
            return False
        ensemble = getattr(info, 'ensemble', None)
        if not ensemble:
            return False
        nodes = [x.split(':', 1)[1] for x in ensemble]
        current_ensemble_nodes = set(nodes) if ensemble else set()
        requested_nodes = set(self.nodes) if self.nodes else set()
        extra_ensemble_nodes = current_ensemble_nodes - requested_nodes
        if extra_ensemble_nodes and self.fail_if_cluster_already_exists_with_larger_ensemble:
            msg = 'Error: found existing cluster with more nodes in ensemble.  Cluster: %s, extra nodes: %s' % (getattr(info, 'cluster', 'not found'), extra_ensemble_nodes)
            msg += '.  Cluster info: %s' % repr(info)
            self.module.fail_json(msg=msg)
        if extra_ensemble_nodes:
            self.debug.append('Extra ensemble nodes: %s' % extra_ensemble_nodes)
        nodes_not_in_ensemble = requested_nodes - current_ensemble_nodes
        if nodes_not_in_ensemble:
            self.debug.append('Extra requested nodes not in ensemble: %s' % nodes_not_in_ensemble)
        return True

    def create_cluster_api(self, options):
        """ Call send_request directly rather than using the SDK if new fields are present
            The new SDK will support these in version 1.17 (Nov or Feb)
        """
        extra_options = ['enableSoftwareEncryptionAtRest', 'orderNumber', 'serialNumber']
        if not any((item in options for item in extra_options)):
            return self.sfe_cluster.create_cluster(**options)
        params = {'mvip': options['mvip'], 'svip': options['svip'], 'repCount': options['rep_count'], 'username': options['username'], 'password': options['password'], 'nodes': options['nodes']}
        if options['accept_eula'] is not None:
            params['acceptEula'] = options['accept_eula']
        if options['attributes'] is not None:
            params['attributes'] = options['attributes']
        for option in extra_options:
            if options.get(option):
                params[option] = options[option]
        return self.sfe_cluster.send_request('CreateCluster', netapp_utils.solidfire.CreateClusterResult, params, since=None)

    def create_cluster(self):
        """
        Create Cluster
        """
        options = {'mvip': self.management_virtual_ip, 'svip': self.storage_virtual_ip, 'rep_count': self.replica_count, 'accept_eula': self.accept_eula, 'nodes': self.nodes, 'attributes': self.attributes, 'username': self.cluster_admin_username, 'password': self.cluster_admin_password}
        if self.encryption is not None:
            options['enableSoftwareEncryptionAtRest'] = self.encryption
        if self.order_number is not None:
            options['orderNumber'] = self.order_number
        if self.serial_number is not None:
            options['serialNumber'] = self.serial_number
        return_msg = 'created'
        try:
            self.create_cluster_api(options)
        except netapp_utils.solidfire.common.ApiServerError as exc:
            if 'xClusterAlreadyCreated' not in str(exc.message):
                self.module.fail_json(msg='Error creating cluster %s' % to_native(exc), exception=traceback.format_exc())
            return_msg = 'already_exists: %s' % str(exc.message)
        except Exception as exc:
            self.module.fail_json(msg='Error creating cluster %s' % to_native(exc), exception=traceback.format_exc())
        return return_msg

    def apply(self):
        """
        Check connection and initialize node with cluster ownership
        """
        changed = False
        result_message = None
        exists = self.check_cluster_exists()
        if exists:
            result_message = 'cluster already exists'
        else:
            changed = True
            if not self.module.check_mode:
                result_message = self.create_cluster()
                if result_message.startswith('already_exists:'):
                    changed = False
        self.module.exit_json(changed=changed, msg=result_message, debug=self.debug)