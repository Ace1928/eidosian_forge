from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
class ACIModule(object):

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.result = dict(changed=False)
        self.headers = dict()
        self.child_classes = set()
        self.connection = None
        self.error = dict(code=None, text=None)
        self.existing = None
        self.config = dict()
        self.original = None
        self.proposed = dict()
        self.stdout = None
        self.filter_string = ''
        self.obj_filter = None
        self.method = None
        self.path = None
        self.parent_path = None
        self.response = None
        self.status = None
        self.url = None
        self.httpapi_logs = list()
        self.imdata = None
        self.totalCount = None
        self.define_protocol()
        self.set_connection()
        if self.module._debug:
            self.module.warn('Enable debug output because ANSIBLE_DEBUG was set.')
            self.params['output_level'] = 'debug'
        if self.params.get('port') is not None:
            self.base_url = '{protocol}://{host}:{port}'.format_map(self.params)
        else:
            self.base_url = '{protocol}://{host}'.format_map(self.params)
        if self.params.get('private_key'):
            if not HAS_CRYPTOGRAPHY and (not HAS_OPENSSL):
                self.module.fail_json(msg='Cannot use signature-based authentication because cryptography (preferred) or pyopenssl are not available')
            elif self.params.get('password') is not None:
                self.module.warn("When doing ACI signatured-based authentication, providing parameter 'password' is not required")
        elif self.connection is None:
            if self.params.get('password'):
                self.login()
            else:
                self.module.fail_json(msg="Either parameter 'password' or 'private_key' is required for authentication")

    def boolean(self, value, true='yes', false='no'):
        """Return an acceptable value back"""
        if value is None:
            return None
        elif value is True:
            return true
        elif value is False:
            return false
        self.module.fail_json(msg="Boolean value '%s' is an invalid ACI boolean value.")

    def iso8601_format(self, dt):
        """Return an ACI-compatible ISO8601 formatted time: 2123-12-12T00:00:00.000+00:00"""
        try:
            return dt.isoformat(timespec='milliseconds')
        except Exception:
            tz = dt.strftime('%z')
            return '%s.%03d%s:%s' % (dt.strftime('%Y-%m-%dT%H:%M:%S'), dt.microsecond / 1000, tz[:3], tz[3:])

    def define_protocol(self):
        """Set protocol based on use_ssl parameter"""
        self.params['protocol'] = 'https' if self.params.get('use_ssl') or self.params.get('use_ssl') is None else 'http'

    def set_connection(self):
        if self.connection is None and self.module._socket_path:
            self.connection = Connection(self.module._socket_path)

    def login(self):
        """Log in to APIC"""
        url = '{0}/api/aaaLogin.json'.format(self.base_url)
        payload = {'aaaUser': {'attributes': {'name': 'admin' if self.params.get('username') is None else self.params.get('username'), 'pwd': self.params.get('password')}}}
        resp, auth = self.api_call('POST', url, data=json.dumps(payload), return_response=True)
        if auth.get('status') != 200:
            self.response = auth.get('msg')
            self.status = auth.get('status')
            try:
                self.response_json(auth['body'])
                self.fail_json(msg='Authentication failed: {code} {text}'.format_map(self.error))
            except KeyError:
                self.fail_json(msg='Connection failed for {url}. {msg}'.format_map(auth))
        self.headers['Cookie'] = resp.headers.get('Set-Cookie')

    def cert_auth(self, path=None, payload='', method=None):
        """Perform APIC signature-based authentication, not the expected SSL client certificate authentication."""
        if method is None:
            method = self.params.get('method').upper()
        if path is None:
            path = self.path
        path = '/' + path.lstrip('/')
        if payload is None:
            payload = ''
        try:
            if HAS_CRYPTOGRAPHY:
                key = self.params.get('private_key').encode()
                sig_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
            else:
                sig_key = load_privatekey(FILETYPE_PEM, self.params.get('private_key'))
        except Exception:
            if os.path.exists(self.params.get('private_key')):
                try:
                    permission = 'r'
                    if HAS_CRYPTOGRAPHY:
                        permission = 'rb'
                    with open(self.params.get('private_key'), permission) as fh:
                        private_key_content = fh.read()
                except Exception:
                    self.module.fail_json(msg="Cannot open private key file '{private_key}'.".format_map(self.params))
                try:
                    if HAS_CRYPTOGRAPHY:
                        sig_key = serialization.load_pem_private_key(private_key_content, password=None, backend=default_backend())
                    else:
                        sig_key = load_privatekey(FILETYPE_PEM, private_key_content)
                except Exception:
                    self.module.fail_json(msg="Cannot load private key file '{private_key}'.".format_map(self.params))
                if self.params.get('certificate_name') is None:
                    self.params['certificate_name'] = os.path.basename(os.path.splitext(self.params.get('private_key'))[0])
            else:
                self.module.fail_json(msg='Provided private key {private_key} does not appear to be a private key or provided file does not exist.'.format_map(self.params))
        if self.params.get('certificate_name') is None:
            self.params['certificate_name'] = 'admin' if self.params.get('username') is None else self.params.get('username')
        sig_request = method + path + payload
        if HAS_CRYPTOGRAPHY:
            sig_signature = sig_key.sign(sig_request.encode(), padding.PKCS1v15(), hashes.SHA256())
        else:
            sig_signature = sign(sig_key, sig_request, 'sha256')
        sig_dn = 'uni/userext/user-{username}/usercert-{certificate_name}'.format_map(self.params)
        self.headers['Cookie'] = 'APIC-Certificate-Algorithm=v1.0; ' + 'APIC-Certificate-DN={0}; '.format(sig_dn) + 'APIC-Certificate-Fingerprint=fingerprint; ' + 'APIC-Request-Signature={0}'.format(to_native(base64.b64encode(sig_signature)))

    def response_json(self, rawoutput):
        """Handle APIC JSON response output"""
        try:
            jsondata = json.loads(rawoutput)
        except Exception as e:
            self.error = dict(code=-1, text="Unable to parse output as JSON, see 'raw' output. {0}".format(e))
            self.result['raw'] = rawoutput
            return
        self.imdata = jsondata.get('imdata')
        if self.imdata is None:
            self.imdata = dict()
        self.totalCount = int(jsondata.get('totalCount'))
        self.response_error()

    def response_xml(self, rawoutput):
        """Handle APIC XML response output"""
        try:
            xml = lxml.etree.fromstring(to_bytes(rawoutput))
            xmldata = cobra.data(xml)
        except Exception as e:
            self.error = dict(code=-1, text="Unable to parse output as XML, see 'raw' output. {0}".format(e))
            self.result['raw'] = rawoutput
            return
        self.imdata = xmldata.get('imdata', {}).get('children')
        if self.imdata is None:
            self.imdata = dict()
        self.totalCount = int(xmldata.get('imdata', {}).get('attributes', {}).get('totalCount', -1))
        self.response_error()

    def response_error(self):
        """Set error information when found"""
        if self.totalCount != '0':
            try:
                self.error = self.imdata[0].get('error').get('attributes')
            except (AttributeError, IndexError, KeyError):
                pass

    def update_qs(self, params):
        """Append key-value pairs to self.filter_string"""
        accepted_params = dict(((k, v) for k, v in params.items() if v is not None))
        if accepted_params:
            if self.filter_string:
                self.filter_string += '&'
            else:
                self.filter_string = '?'
            self.filter_string += '&'.join(['%s=%s' % (k, v) for k, v in accepted_params.items()])

    def build_filter(self, obj_class, params):
        """Build an APIC filter based on obj_class and key-value pairs"""
        accepted_params = dict(((k, v) for k, v in params.items() if v is not None))
        if len(accepted_params) == 1:
            return ','.join(('eq({0}.{1},"{2}")'.format(obj_class, k, v) for k, v in accepted_params.items()))
        elif len(accepted_params) > 1:
            return 'and(' + ','.join(['eq({0}.{1},"{2}")'.format(obj_class, k, v) for k, v in accepted_params.items()]) + ')'

    def _deep_url_path_builder(self, obj):
        target_class = obj.get('target_class')
        target_filter = obj.get('target_filter')
        subtree_class = obj.get('subtree_class')
        subtree_filter = obj.get('subtree_filter')
        object_rn = obj.get('object_rn')
        mo = obj.get('module_object')
        add_subtree_filter = obj.get('add_subtree_filter')
        add_target_filter = obj.get('add_target_filter')
        if self.module.params.get('state') in ('absent', 'present') and mo is not None:
            self.path = 'api/mo/uni/{0}.json'.format(object_rn)
            self.update_qs({'rsp-prop-include': 'config-only'})
        else:
            if object_rn is not None:
                self.path = 'api/mo/uni/{0}.json'.format(object_rn)
            else:
                self.path = 'api/class/{0}.json'.format(target_class)
            if add_target_filter:
                self.update_qs({'query-target-filter': self.build_filter(target_class, target_filter)})
            if add_subtree_filter:
                self.update_qs({'rsp-subtree-filter': self.build_filter(subtree_class, subtree_filter)})
        if self.params.get('port') is not None:
            self.url = '{protocol}://{host}:{port}/{path}'.format(path=self.path, **self.module.params)
        else:
            self.url = '{protocol}://{host}/{path}'.format(path=self.path, **self.module.params)
        if self.child_classes:
            self.update_qs({'rsp-subtree': 'full', 'rsp-subtree-class': ','.join(sorted(self.child_classes))})

    def _deep_url_parent_object(self, parent_objects, parent_class):
        for parent_object in parent_objects:
            if parent_object.get('aci_class') is parent_class:
                return parent_object
        return None

    def construct_deep_url(self, target_object, parent_objects=None, child_classes=None):
        """
        This method is used to retrieve the appropriate URL path and filter_string to make the request to the APIC.

        :param target_object: The target class dictionary containing parent_class, aci_class, aci_rn, target_filter, and module_object keys.
        :param parent_objects: The parent class list of dictionaries containing parent_class, aci_class, aci_rn, target_filter, and module_object keys.
        :param child_classes: The list of child classes that the module supports along with the object.
        :type target_object: dict
        :type parent_objects: list[dict]
        :type child_classes: list[string]
        :return: The path and filter_string needed to build the full URL.
        """
        self.filter_string = ''
        rn_builder = None
        subtree_classes = None
        add_subtree_filter = False
        add_target_filter = False
        has_target_query = False
        has_target_query_compare = False
        has_target_query_difference = False
        has_target_query_called = False
        if child_classes is None:
            self.child_classes = set()
        else:
            self.child_classes = set(child_classes)
        target_parent_class = target_object.get('parent_class')
        target_class = target_object.get('aci_class')
        target_rn = target_object.get('aci_rn')
        target_filter = target_object.get('target_filter')
        target_module_object = target_object.get('module_object')
        url_path_object = dict(target_class=target_class, target_filter=target_filter, subtree_class=target_class, subtree_filter=target_filter, module_object=target_module_object)
        if target_module_object is not None:
            rn_builder = target_rn
        else:
            has_target_query = True
            has_target_query_compare = True
        if parent_objects is not None:
            current_parent_class = target_parent_class
            has_parent_query_compare = False
            has_parent_query_difference = False
            is_first_parent = True
            is_single_parent = None
            search_classes = set()
            while current_parent_class != 'uni':
                parent_object = self._deep_url_parent_object(parent_objects=parent_objects, parent_class=current_parent_class)
                if parent_object is not None:
                    parent_parent_class = parent_object.get('parent_class')
                    parent_class = parent_object.get('aci_class')
                    parent_rn = parent_object.get('aci_rn')
                    parent_filter = parent_object.get('target_filter')
                    parent_module_object = parent_object.get('module_object')
                    if is_first_parent:
                        is_single_parent = True
                    else:
                        is_single_parent = False
                    is_first_parent = False
                    if parent_parent_class != 'uni':
                        search_classes.add(parent_class)
                    if parent_module_object is not None:
                        if rn_builder is not None:
                            rn_builder = '{0}/{1}'.format(parent_rn, rn_builder)
                        else:
                            rn_builder = parent_rn
                        url_path_object['target_class'] = parent_class
                        url_path_object['target_filter'] = parent_filter
                        has_target_query = False
                    else:
                        rn_builder = None
                        subtree_classes = search_classes
                        has_target_query = True
                        if is_single_parent:
                            has_parent_query_compare = True
                    current_parent_class = parent_parent_class
                else:
                    raise ValueError("Reference error for parent_class '{0}'. Each parent_class must reference a valid object".format(current_parent_class))
                if not has_target_query_difference and (not has_target_query_called):
                    if has_target_query is not has_target_query_compare:
                        has_target_query_difference = True
                elif not has_parent_query_difference and has_target_query is not has_parent_query_compare:
                    has_parent_query_difference = True
                has_target_query_called = True
            if not has_parent_query_difference and has_parent_query_compare and (target_module_object is not None):
                add_target_filter = True
            elif has_parent_query_difference and target_module_object is not None:
                add_subtree_filter = True
                self.child_classes.add(target_class)
                if has_target_query:
                    add_target_filter = True
            elif has_parent_query_difference and (not has_target_query) and (target_module_object is None):
                self.child_classes.add(target_class)
                self.child_classes.update(subtree_classes)
            elif not has_parent_query_difference and (not has_target_query) and (target_module_object is None):
                self.child_classes.add(target_class)
            elif not has_target_query and is_single_parent and (target_module_object is None):
                self.child_classes.add(target_class)
        url_path_object['object_rn'] = rn_builder
        url_path_object['add_subtree_filter'] = add_subtree_filter
        url_path_object['add_target_filter'] = add_target_filter
        self._deep_url_path_builder(url_path_object)

    def construct_url(self, root_class, subclass_1=None, subclass_2=None, subclass_3=None, subclass_4=None, subclass_5=None, child_classes=None, config_only=True):
        """
        This method is used to retrieve the appropriate URL path and filter_string to make the request to the APIC.

        :param root_class: The top-level class dictionary containing aci_class, aci_rn, target_filter, and module_object keys.
        :param sublass_1: The second-level class dictionary containing aci_class, aci_rn, target_filter, and module_object keys.
        :param sublass_2: The third-level class dictionary containing aci_class, aci_rn, target_filter, and module_object keys.
        :param sublass_3: The fourth-level class dictionary containing aci_class, aci_rn, target_filter, and module_object keys.
        :param child_classes: The list of child classes that the module supports along with the object.
        :type root_class: dict
        :type subclass_1: dict
        :type subclass_2: dict
        :type subclass_3: dict
        :type subclass_4: dict
        :type subclass_5: dict
        :type child_classes: list
        :return: The path and filter_string needed to build the full URL.
        """
        self.filter_string = ''
        if child_classes is None:
            self.child_classes = set()
        else:
            self.child_classes = set(child_classes)
        if subclass_5 is not None:
            self._construct_url_6(root_class, subclass_1, subclass_2, subclass_3, subclass_4, subclass_5, config_only)
        elif subclass_4 is not None:
            self._construct_url_5(root_class, subclass_1, subclass_2, subclass_3, subclass_4, config_only)
        elif subclass_3 is not None:
            self._construct_url_4(root_class, subclass_1, subclass_2, subclass_3, config_only)
        elif subclass_2 is not None:
            self._construct_url_3(root_class, subclass_1, subclass_2, config_only)
        elif subclass_1 is not None:
            self._construct_url_2(root_class, subclass_1, config_only)
        else:
            self._construct_url_1(root_class, config_only)
        if self.params.get('port') is not None:
            self.url = '{protocol}://{host}:{port}/{path}'.format(path=self.path, **self.module.params)
        else:
            self.url = '{protocol}://{host}/{path}'.format(path=self.path, **self.module.params)
        if self.child_classes:
            self.update_qs({'rsp-subtree': 'full', 'rsp-subtree-class': ','.join(sorted(self.child_classes))})

    def _construct_url_1(self, obj, config_only=True):
        """
        This method is used by construct_url when the object is the top-level class.
        """
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}.json'.format(obj_rn)
            self.parent_path = 'api/mo/uni.json'
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif mo is None:
            self.path = 'api/class/{0}.json'.format(obj_class)
            if obj_filter is not None:
                self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        else:
            self.path = 'api/mo/uni/{0}.json'.format(obj_rn)

    def _construct_url_2(self, parent, obj, config_only=True):
        """
        This method is used by construct_url when the object is the second-level class.
        """
        parent_rn = parent.get('aci_rn')
        parent_obj = parent.get('module_object')
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}/{1}.json'.format(parent_rn, obj_rn)
            self.parent_path = 'api/mo/uni/{0}.json'.format(parent_rn)
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif parent_obj is None and mo is None:
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif parent_obj is None:
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}.json'.format(parent_rn)
        else:
            self.path = 'api/mo/uni/{0}/{1}.json'.format(parent_rn, obj_rn)

    def _construct_url_3(self, root, parent, obj, config_only=True):
        """
        This method is used by construct_url when the object is the third-level class.
        """
        root_rn = root.get('aci_rn')
        root_obj = root.get('module_object')
        parent_class = parent.get('aci_class')
        parent_rn = parent.get('aci_rn')
        parent_filter = parent.get('target_filter')
        parent_obj = parent.get('module_object')
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, parent_rn, obj_rn)
            self.parent_path = 'api/mo/uni/{0}/{1}.json'.format(root_rn, parent_rn)
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif root_obj is None and parent_obj is None and (mo is None):
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif root_obj is None and parent_obj is None:
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif root_obj is None and mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/class/{0}.json'.format(parent_class)
            self.update_qs({'query-target-filter': self.build_filter(parent_class, parent_filter)})
        elif parent_obj is None and mo is None:
            self.child_classes.update([parent_class, obj_class])
            self.path = 'api/mo/uni/{0}.json'.format(root_rn)
        elif root_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/class/{0}.json'.format(parent_class)
            self.update_qs({'query-target-filter': self.build_filter(parent_class, parent_filter)})
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif parent_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}.json'.format(root_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}.json'.format(root_rn, parent_rn)
        else:
            self.path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, parent_rn, obj_rn)

    def _construct_url_4(self, root, sec, parent, obj, config_only=True):
        """
        This method is used by construct_url when the object is the fourth-level class.
        """
        root_rn = root.get('aci_rn')
        root_obj = root.get('module_object')
        sec_rn = sec.get('aci_rn')
        sec_obj = sec.get('module_object')
        parent_rn = parent.get('aci_rn')
        parent_obj = parent.get('module_object')
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.child_classes is None:
            self.child_classes = [obj_class]
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}.json'.format(root_rn, sec_rn, parent_rn, obj_rn)
            self.parent_path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, sec_rn, parent_rn)
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif root_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif sec_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}.json'.format(root_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif parent_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}.json'.format(root_rn, sec_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, sec_rn, parent_rn)
        else:
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}.json'.format(root_rn, sec_rn, parent_rn, obj_rn)

    def _construct_url_5(self, root, ter, sec, parent, obj, config_only=True):
        """
        This method is used by construct_url when the object is the fourth-level class.
        """
        root_rn = root.get('aci_rn')
        root_obj = root.get('module_object')
        ter_rn = ter.get('aci_rn')
        ter_obj = ter.get('module_object')
        sec_rn = sec.get('aci_rn')
        sec_obj = sec.get('module_object')
        parent_rn = parent.get('aci_rn')
        parent_obj = parent.get('module_object')
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.child_classes is None:
            self.child_classes = [obj_class]
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}/{4}.json'.format(root_rn, ter_rn, sec_rn, parent_rn, obj_rn)
            self.parent_path = 'api/mo/uni/{0}/{1}/{2}/{3}.json'.format(root_rn, ter_rn, sec_rn, parent_rn)
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif root_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif ter_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}.json'.format(root_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif sec_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}.json'.format(root_rn, ter_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif parent_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, ter_rn, sec_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}.json'.format(root_rn, ter_rn, sec_rn, parent_rn)
        else:
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}/{4}.json'.format(root_rn, ter_rn, sec_rn, parent_rn, obj_rn)

    def _construct_url_6(self, root, quad, ter, sec, parent, obj, config_only=True):
        """
        This method is used by construct_url when the object is the fourth-level class.
        """
        root_rn = root.get('aci_rn')
        root_obj = root.get('module_object')
        quad_rn = quad.get('aci_rn')
        quad_obj = quad.get('module_object')
        ter_rn = ter.get('aci_rn')
        ter_obj = ter.get('module_object')
        sec_rn = sec.get('aci_rn')
        sec_obj = sec.get('module_object')
        parent_rn = parent.get('aci_rn')
        parent_obj = parent.get('module_object')
        obj_class = obj.get('aci_class')
        obj_rn = obj.get('aci_rn')
        obj_filter = obj.get('target_filter')
        mo = obj.get('module_object')
        if self.child_classes is None:
            self.child_classes = [obj_class]
        if self.module.params.get('state') in ('absent', 'present'):
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}/{4}/{5}.json'.format(root_rn, quad_rn, ter_rn, sec_rn, parent_rn, obj_rn)
            if config_only:
                self.update_qs({'rsp-prop-include': 'config-only'})
            self.obj_filter = obj_filter
        elif root_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/class/{0}.json'.format(obj_class)
            self.update_qs({'query-target-filter': self.build_filter(obj_class, obj_filter)})
        elif quad_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}.json'.format(root_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif ter_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}.json'.format(root_rn, quad_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif sec_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}.json'.format(root_rn, quad_rn, ter_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif parent_obj is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}.json'.format(root_rn, quad_rn, ter_rn, sec_rn)
            self.update_qs({'rsp-subtree-filter': self.build_filter(obj_class, obj_filter)})
        elif mo is None:
            self.child_classes.add(obj_class)
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}/{4}.json'.format(root_rn, quad_rn, ter_rn, sec_rn, parent_rn)
        else:
            self.path = 'api/mo/uni/{0}/{1}/{2}/{3}/{4}/{5}.json'.format(root_rn, quad_rn, ter_rn, sec_rn, parent_rn, obj_rn)

    def delete_config(self):
        """
        This method is used to handle the logic when the modules state is equal to absent. The method only pushes a change if
        the object exists, and if check_mode is False. A successful change will mark the module as changed.
        """
        self.proposed = dict()
        if not self.existing:
            return
        elif not self.module.check_mode:
            self.api_call('DELETE', self.url, None, return_response=False)
        else:
            self.result['changed'] = True
            self.method = 'DELETE'

    def get_diff(self, aci_class):
        """
        This method is used to get the difference between the proposed and existing configurations. Each module
        should call the get_existing method before this method, and add the proposed config to the module results
        using the module's config parameters. The new config will added to the self.result dictionary.
        :param aci_class: Type str.
                          This is the root dictionary key for the MO's configuration body, or the ACI class of the MO.
        """
        proposed_config = self.proposed[aci_class]['attributes']
        if self.existing:
            existing_config = self.existing[0][aci_class]['attributes']
            config = {}
            for key, value in proposed_config.items():
                existing_field = existing_config.get(key)
                if value != existing_field:
                    config[key] = value
            if config:
                config = {aci_class: {'attributes': config}}
            children = self.get_diff_children(aci_class)
            if children and config:
                config[aci_class].update({'children': children})
            elif children:
                config = {aci_class: {'attributes': {}, 'children': children}}
        else:
            config = self.proposed
        self.config = config

    @staticmethod
    def get_diff_child(child_class, proposed_child, existing_child):
        """
        This method is used to get the difference between a proposed and existing child configs. The get_nested_config()
        method should be used to return the proposed and existing config portions of child.
        :param child_class: Type str.
                            The root class (dict key) for the child dictionary.
        :param proposed_child: Type dict.
                               The config portion of the proposed child dictionary.
        :param existing_child: Type dict.
                               The config portion of the existing child dictionary.
        :return: The child config with only values that are updated. If the proposed dictionary has no updates to make
                 to what exists on the APIC, then None is returned.
        """
        update_config = {child_class: {'attributes': {}}}
        for key, value in proposed_child.items():
            existing_field = existing_child.get(key)
            if value != existing_field:
                update_config[child_class]['attributes'][key] = value
        if not update_config[child_class]['attributes']:
            return None
        return update_config

    def get_diff_children(self, aci_class, proposed_obj=None, existing_obj=None):
        """
        This method is used to retrieve the updated child configs by comparing the proposed children configs
        against the objects existing children configs.
        :param aci_class: Type str.
                          This is the root dictionary key for the MO's configuration body, or the ACI class of the MO.
        :return: The list of updated child config dictionaries. None is returned if there are no changes to the child
                 configurations.
        """
        if proposed_obj is None:
            proposed_children = self.proposed[aci_class].get('children')
        else:
            proposed_children = proposed_obj
        if proposed_children:
            child_updates = []
            if existing_obj is None:
                existing_children = self.existing[0][aci_class].get('children', [])
            else:
                existing_children = existing_obj
            for child in proposed_children:
                child_class, proposed_child, existing_child = self.get_nested_config(child, existing_children)
                proposed_child_children, existing_child_children = self.get_nested_children(child, existing_children)
                if existing_child is None:
                    child_update = child
                else:
                    child_update = self.get_diff_child(child_class, proposed_child, existing_child)
                    if proposed_child_children:
                        child_update_children = self.get_diff_children(aci_class, proposed_child_children, existing_child_children)
                        if child_update_children:
                            child_update = child
                if child_update:
                    child_updates.append(child_update)
        else:
            return None
        return child_updates

    def get_existing(self):
        """
        This method is used to get the existing object(s) based on the path specified in the module. Each module should
        build the URL so that if the object's name is supplied, then it will retrieve the configuration for that particular
        object, but if no name is supplied, then it will retrieve all MOs for the class. Following this method will ensure
        that this method can be used to supply the existing configuration when using the get_diff method. The response, status,
        and existing configuration will be added to the self.result dictionary.
        """
        uri = self.url + self.filter_string
        self.api_call('GET', uri, data=None, return_response=False)

    @staticmethod
    def get_nested_config(proposed_child, existing_children):
        """
        This method is used for stiping off the outer layers of the child dictionaries so only the configuration
        key, value pairs are returned.
        :param proposed_child: Type dict.
                               The dictionary that represents the child config.
        :param existing_children: Type list.
                                  The list of existing child config dictionaries.
        :return: The child's class as str (root config dict key), the child's proposed config dict, and the child's
                 existing configuration dict.
        """
        for key in proposed_child.keys():
            child_class = key
            proposed_config = proposed_child[key]['attributes']
            existing_config = None
            for child in existing_children:
                if child.get(child_class):
                    existing_config = child[key]['attributes']
                    if set(proposed_config.items()).issubset(set(existing_config.items())):
                        break
                    existing_config = None
        return (child_class, proposed_config, existing_config)

    @staticmethod
    def get_nested_children(proposed_child, existing_children):
        """
        This method is used for stiping off the outer layers of the child dictionaries so only the children are returned.
        :param proposed_child: Type dict.
                               The dictionary that represents the child config.
        :param existing_children: Type list.
                                  The list of existing child config dictionaries.
        :return: The child's class as str (root config dict key), the child's proposed children as a list and the child's
                 existing children as a list.
        """
        for key in proposed_child.keys():
            child_class = key
            proposed_config = proposed_child[key]['attributes']
            existing_config = None
            proposed_children = proposed_child[key].get('children')
            existing_child_children = None
            for child in existing_children:
                if child.get(child_class):
                    existing_config = child[key]['attributes']
                    existing_child_children = child[key].get('children')
                    if set(proposed_config.items()).issubset(set(existing_config.items())):
                        break
                    existing_child_children = None
                    existing_config = None
        return (proposed_children, existing_child_children)

    def payload(self, aci_class, class_config, child_configs=None):
        """
        This method is used to dynamically build the proposed configuration dictionary from the config related parameters
        passed into the module. All values that were not passed values from the playbook task will be removed so as to not
        inadvertently change configurations.
        :param aci_class: Type str
                          This is the root dictionary key for the MO's configuration body, or the ACI class of the MO.
        :param class_config: Type dict
                             This is the configuration of the MO using the dictionary keys expected by the API
        :param child_configs: Type list
                              This is a list of child dictionaries associated with the MOs config. The list should only
                              include child objects that are used to associate two MOs together. Children that represent
                              MOs should have their own module.
        """
        proposed = dict(((k, str(v)) for k, v in class_config.items() if v is not None))
        if self.params.get('annotation') is not None:
            proposed['annotation'] = self.params.get('annotation')
        if self.params.get('owner_key') is not None:
            proposed['ownerKey'] = self.params.get('owner_key')
        if self.params.get('owner_tag') is not None:
            proposed['ownerTag'] = self.params.get('owner_tag')
        self.proposed = {aci_class: {'attributes': proposed}}
        if child_configs:
            children = []
            for child in child_configs:
                child_copy = deepcopy(child)
                has_value = False
                for root_key in child_copy.keys():
                    for final_keys, values in child_copy[root_key]['attributes'].items():
                        if values is None:
                            child[root_key]['attributes'].pop(final_keys)
                        else:
                            child[root_key]['attributes'][final_keys] = str(values)
                            has_value = True
                if has_value:
                    children.append(child)
            if children:
                self.proposed[aci_class].update(dict(children=children))

    def post_config(self, parent_class=None):
        """
        This method is used to handle the logic when the modules state is equal to present. The method only pushes a change if
        the object has differences than what exists on the APIC, and if check_mode is False. A successful change will mark the
        module as changed.
        """
        if not self.config:
            return
        elif not self.module.check_mode:
            url = self.url
            if parent_class is not None:
                if self.params.get('port') is not None:
                    url = '{protocol}://{host}:{port}/{path}'.format(path=self.parent_path, **self.module.params)
                else:
                    url = '{protocol}://{host}/{path}'.format(path=self.parent_path, **self.module.params)
                self.config = {parent_class: {'attributes': {}, 'children': [self.config]}}
            self.api_call('POST', url, json.dumps(self.config), return_response=False)
        else:
            self.result['changed'] = True
            self.method = 'POST'

    def exit_json(self, filter_existing=None, **kwargs):
        """
        :param filter_existing: tuple consisting of the function at (index 0) and the args at (index 1)
        CAUTION: the function should always take in self.existing in its first parameter
        :param kwargs: kwargs to be passed to ansible module exit_json()
        filter_existing is not passed via kwargs since it cant handle function type and should not be exposed to user
        """
        if 'state' in self.params:
            if self.params.get('state') in ('absent', 'present'):
                if self.params.get('output_level') in ('debug', 'info'):
                    self.result['previous'] = self.existing if not filter_existing else filter_existing[0](self.existing, filter_existing[1])
        if self.params.get('output_level') == 'debug':
            if 'state' in self.params:
                self.result['filter_string'] = self.filter_string
            self.result['method'] = self.method
            self.result['response'] = self.response
            self.result['status'] = self.status
            self.result['url'] = self.url
            if self.httpapi_logs is not None:
                self.result['httpapi_logs'] = self.httpapi_logs
        if self.stdout:
            self.result['stdout'] = self.stdout
        if 'state' in self.params:
            self.original = self.existing
            if self.params.get('state') in ('absent', 'present'):
                self.get_existing()
            self.result['current'] = self.existing if not filter_existing else filter_existing[0](self.existing, filter_existing[1])
            if self.params.get('output_level') in ('debug', 'info'):
                self.result['sent'] = self.config
                self.result['proposed'] = self.proposed
        self.dump_json()
        self.result.update(**kwargs)
        self.module.exit_json(**self.result)

    def fail_json(self, msg, **kwargs):
        if self.error.get('code') is not None and self.error.get('text') is not None:
            self.result['error'] = self.error
        if self.stdout:
            self.result['stdout'] = self.stdout
        if 'state' in self.params:
            if self.params.get('state') in ('absent', 'present'):
                if self.params.get('output_level') in ('debug', 'info'):
                    self.result['previous'] = self.existing
            if self.params.get('output_level') == 'debug':
                if self.imdata is not None:
                    self.result['imdata'] = self.imdata
                    self.result['totalCount'] = self.totalCount
        if self.params.get('output_level') == 'debug':
            if self.url is not None:
                if 'state' in self.params:
                    self.result['filter_string'] = self.filter_string
                self.result['method'] = self.method
                self.result['response'] = self.response
                self.result['status'] = self.status
                self.result['url'] = self.url
            if self.httpapi_logs is not None:
                self.result['httpapi_logs'] = self.httpapi_logs
        if 'state' in self.params:
            if self.params.get('output_level') in ('debug', 'info'):
                self.result['sent'] = self.config
                self.result['proposed'] = self.proposed
        self.result.update(**kwargs)
        self.module.fail_json(msg=msg, **self.result)

    def dump_json(self):
        if self.params.get('state') in ('absent', 'present'):
            dn_path = self.url.rsplit('/mo/', maxsplit=1)[-1]
            if dn_path[-5:] == '.json':
                dn_path = dn_path[:-5]
            mo = {}
            if self.proposed:
                mo = self.proposed
                for aci_class in mo:
                    mo[aci_class]['attributes']['dn'] = dn_path
                    if self.obj_filter is not None:
                        if 'tDn' in self.obj_filter:
                            mo[aci_class]['attributes']['tDn'] = self.obj_filter['tDn']
            elif self.params.get('state') == 'absent' and self.existing:
                for aci_class in self.existing[0]:
                    mo[aci_class] = dict(attributes=dict(dn=dn_path, status='deleted'))
            self.result['mo'] = mo
            output_path = self.params.get('output_path')
            if output_path is not None:
                with open(output_path, 'a') as output_file:
                    if self.result.get('changed') is True:
                        json.dump([mo], output_file)

    def parsed_url_path(self, url):
        if not HAS_URLPARSE:
            self.fail_json(msg='urlparse is not installed')
        parse_result = urlparse(url)
        if parse_result.query == '':
            return parse_result.path
        else:
            return parse_result.path + '?' + parse_result.query

    def api_call(self, method, url, data=None, return_response=False):
        resp = None
        if self.connection is not None:
            self.connection.set_params(self.params)
            info = self.connection.send_request(method, self.parsed_url_path(url), data)
            self.url = '{protocol}://{host}/{path}'.format_map(integrate_url(info.get('url'), self.path))
            self.error = info.get('error')
            self.httpapi_logs.extend(self.connection.pop_messages())
        else:
            if self.params.get('private_key'):
                self.cert_auth(path=self.parsed_url_path(url), payload=data, method=method)
            resp, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method, timeout=30 if self.params.get('timeout') is None else self.params.get('timeout'), use_proxy=True if self.params.get('use_proxy') is None else self.params.get('use_proxy'))
        self.response = info.get('msg')
        self.status = info.get('status')
        self.method = method
        if return_response:
            return (resp, info)
        elif info.get('status') == 200:
            if method == 'POST' or method == 'DELETE':
                self.result['changed'] = True
            try:
                if method == 'GET':
                    self.existing = json.loads(resp.read())['imdata']
                else:
                    self.response_json(resp.read())
            except AttributeError:
                if method == 'GET':
                    self.existing = json.loads(info.get('body'))['imdata']
                else:
                    self.response_json(info.get('body'))
        else:
            try:
                self.response_json(info['body'])
                self.fail_json(msg='APIC Error {code}: {text}'.format_map(self.error))
            except KeyError:
                self.fail_json(msg='Connection failed for {url}. {msg}'.format_map(info))