from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
class EcsDomain(object):
    """
    Entrust Certificate Services domain class.
    """

    def __init__(self, module):
        self.changed = False
        self.domain_status = None
        self.verification_method = None
        self.file_location = None
        self.file_contents = None
        self.dns_location = None
        self.dns_contents = None
        self.dns_resource_type = None
        self.emails = None
        self.ov_eligible = None
        self.ov_days_remaining = None
        self.ev_eligble = None
        self.ev_days_remaining = None
        self.verification_method = None
        self.ecs_client = None
        try:
            self.ecs_client = ECSClient(entrust_api_user=module.params['entrust_api_user'], entrust_api_key=module.params['entrust_api_key'], entrust_api_cert=module.params['entrust_api_client_cert_path'], entrust_api_cert_key=module.params['entrust_api_client_cert_key_path'], entrust_api_specification_path=module.params['entrust_api_specification_path'])
        except SessionConfigurationException as e:
            module.fail_json(msg='Failed to initialize Entrust Provider: {0}'.format(to_native(e)))
        try:
            self.ecs_client.GetAppVersion()
        except RestOperationException as e:
            module.fail_json(msg='Please verify credential information. Received exception when testing ECS connection: {0}'.format(to_native(e.message)))

    def set_domain_details(self, domain_details):
        if domain_details.get('verificationMethod'):
            self.verification_method = domain_details['verificationMethod'].lower()
        self.domain_status = domain_details['verificationStatus']
        self.ov_eligible = domain_details.get('ovEligible')
        self.ov_days_remaining = calculate_days_remaining(domain_details.get('ovExpiry'))
        self.ev_eligible = domain_details.get('evEligible')
        self.ev_days_remaining = calculate_days_remaining(domain_details.get('evExpiry'))
        self.client_id = domain_details['clientId']
        if self.verification_method == 'dns' and domain_details.get('dnsMethod'):
            self.dns_location = domain_details['dnsMethod']['recordDomain']
            self.dns_resource_type = domain_details['dnsMethod']['recordType']
            self.dns_contents = domain_details['dnsMethod']['recordValue']
        elif self.verification_method == 'web_server' and domain_details.get('webServerMethod'):
            self.file_location = domain_details['webServerMethod']['fileLocation']
            self.file_contents = domain_details['webServerMethod']['fileContents']
        elif self.verification_method == 'email' and domain_details.get('emailMethod'):
            self.emails = domain_details['emailMethod']

    def check(self, module):
        try:
            domain_details = self.ecs_client.GetDomain(clientId=module.params['client_id'], domain=module.params['domain_name'])
            self.set_domain_details(domain_details)
            if self.domain_status != 'APPROVED' and self.domain_status != 'INITIAL_VERIFICATION' and (self.domain_status != 'RE_VERIFICATION'):
                return False
            if self.domain_status == 'INITIAL_VERIFICATION' or self.domain_status == 'RE_VERIFICATION':
                if self.verification_method != module.params['verification_method']:
                    return False
            if self.domain_status == 'EXPIRING':
                return False
            return True
        except RestOperationException as dummy:
            return False

    def request_domain(self, module):
        if not self.check(module):
            body = {}
            body['verificationMethod'] = module.params['verification_method'].upper()
            if module.params['verification_method'] == 'email':
                emailMethod = {}
                if module.params['verification_email']:
                    emailMethod['emailSource'] = 'SPECIFIED'
                    emailMethod['email'] = module.params['verification_email']
                else:
                    emailMethod['emailSource'] = 'INCLUDE_WHOIS'
                body['emailMethod'] = emailMethod
            if not self.domain_status:
                body['domainName'] = module.params['domain_name']
            try:
                if not self.domain_status:
                    self.ecs_client.AddDomain(clientId=module.params['client_id'], Body=body)
                else:
                    self.ecs_client.ReverifyDomain(clientId=module.params['client_id'], domain=module.params['domain_name'], Body=body)
                time.sleep(5)
                result = self.ecs_client.GetDomain(clientId=module.params['client_id'], domain=module.params['domain_name'])
                if module.params['verification_method'] == 'dns' or module.params['verification_method'] == 'web_server':
                    for i in range(4):
                        if module.params['verification_method'] == 'dns':
                            if result.get('dnsMethod') and result['dnsMethod']['recordValue'] != self.dns_contents:
                                break
                        elif module.params['verification_method'] == 'web_server':
                            if result.get('webServerMethod') and result['webServerMethod']['fileContents'] != self.file_contents:
                                break
                    time.sleep(10)
                    result = self.ecs_client.GetDomain(clientId=module.params['client_id'], domain=module.params['domain_name'])
                self.changed = True
                self.set_domain_details(result)
            except RestOperationException as e:
                module.fail_json(msg='Failed to request domain validation from Entrust (ECS) {0}'.format(e.message))

    def dump(self):
        result = {'changed': self.changed, 'client_id': self.client_id, 'domain_status': self.domain_status}
        if self.verification_method:
            result['verification_method'] = self.verification_method
        if self.ov_eligible is not None:
            result['ov_eligible'] = self.ov_eligible
        if self.ov_days_remaining:
            result['ov_days_remaining'] = self.ov_days_remaining
        if self.ev_eligible is not None:
            result['ev_eligible'] = self.ev_eligible
        if self.ev_days_remaining:
            result['ev_days_remaining'] = self.ev_days_remaining
        if self.emails:
            result['emails'] = self.emails
        if self.verification_method == 'dns':
            result['dns_location'] = self.dns_location
            result['dns_contents'] = self.dns_contents
            result['dns_resource_type'] = self.dns_resource_type
        elif self.verification_method == 'web_server':
            result['file_location'] = self.file_location
            result['file_contents'] = self.file_contents
        elif self.verification_method == 'email':
            result['emails'] = self.emails
        return result