from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
from ansible_collections.community.crypto.plugins.module_utils.acme.orders import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
class ACMECertificateClient(object):
    """
    ACME client class. Uses an ACME account object and a CSR to
    start and validate ACME challenges and download the respective
    certificates.
    """

    def __init__(self, module, backend):
        self.module = module
        self.version = module.params['acme_version']
        self.challenge = module.params['challenge']
        if self.challenge == NO_CHALLENGE:
            self.challenge = None
        self.csr = module.params['csr']
        self.csr_content = module.params['csr_content']
        self.dest = module.params.get('dest')
        self.fullchain_dest = module.params.get('fullchain_dest')
        self.chain_dest = module.params.get('chain_dest')
        self.client = ACMEClient(module, backend)
        self.account = ACMEAccount(self.client)
        self.directory = self.client.directory
        self.data = module.params['data']
        self.authorizations = None
        self.cert_days = -1
        self.order = None
        self.order_uri = self.data.get('order_uri') if self.data else None
        self.all_chains = None
        self.select_chain_matcher = []
        if self.module.params['select_chain']:
            for criterium_idx, criterium in enumerate(self.module.params['select_chain']):
                try:
                    self.select_chain_matcher.append(self.client.backend.create_chain_matcher(Criterium(criterium, index=criterium_idx)))
                except ValueError as exc:
                    self.module.warn('Error while parsing criterium: {error}. Ignoring criterium.'.format(error=exc))
        modify_account = module.params['modify_account']
        if modify_account or self.version > 1:
            contact = []
            if module.params['account_email']:
                contact.append('mailto:' + module.params['account_email'])
            created, account_data = self.account.setup_account(contact, agreement=module.params.get('agreement'), terms_agreed=module.params.get('terms_agreed'), allow_creation=modify_account)
            if account_data is None:
                raise ModuleFailException(msg='Account does not exist or is deactivated.')
            updated = False
            if not created and account_data and modify_account:
                updated, account_data = self.account.update_account(account_data, contact)
            self.changed = created or updated
        else:
            pass
        if self.csr is not None and (not os.path.exists(self.csr)):
            raise ModuleFailException('CSR %s not found' % self.csr)
        self.identifiers = self.client.backend.get_csr_identifiers(csr_filename=self.csr, csr_content=self.csr_content)

    def is_first_step(self):
        """
        Return True if this is the first execution of this module, i.e. if a
        sufficient data object from a first run has not been provided.
        """
        if self.data is None:
            return True
        if self.version == 1:
            return not self.data
        else:
            return self.order_uri is None

    def start_challenges(self):
        """
        Create new authorizations for all identifiers of the CSR,
        respectively start a new order for ACME v2.
        """
        self.authorizations = {}
        if self.version == 1:
            for identifier_type, identifier in self.identifiers:
                if identifier_type != 'dns':
                    raise ModuleFailException('ACME v1 only supports DNS identifiers!')
            for identifier_type, identifier in self.identifiers:
                authz = Authorization.create(self.client, identifier_type, identifier)
                self.authorizations[authz.combined_identifier] = authz
        else:
            self.order = Order.create(self.client, self.identifiers)
            self.order_uri = self.order.url
            self.order.load_authorizations(self.client)
            self.authorizations.update(self.order.authorizations)
        self.changed = True

    def get_challenges_data(self, first_step):
        """
        Get challenge details for the chosen challenge type.
        Return a tuple of generic challenge details, and specialized DNS challenge details.
        """
        data = {}
        for type_identifier, authz in self.authorizations.items():
            identifier_type, identifier = split_identifier(type_identifier)
            if authz.status == 'valid':
                continue
            data[identifier] = authz.get_challenge_data(self.client)
            if first_step and self.challenge is not None and (self.challenge not in data[identifier]):
                raise ModuleFailException("Found no challenge of type '{0}' for identifier {1}!".format(self.challenge, type_identifier))
        data_dns = {}
        if self.challenge == 'dns-01':
            for identifier, challenges in data.items():
                if self.challenge in challenges:
                    values = data_dns.get(challenges[self.challenge]['record'])
                    if values is None:
                        values = []
                        data_dns[challenges[self.challenge]['record']] = values
                    values.append(challenges[self.challenge]['resource_value'])
        return (data, data_dns)

    def finish_challenges(self):
        """
        Verify challenges for all identifiers of the CSR.
        """
        self.authorizations = {}
        if self.version == 1:
            for identifier_type, identifier in self.identifiers:
                authz = Authorization.create(self.client, identifier_type, identifier)
                self.authorizations[combine_identifier(identifier_type, identifier)] = authz
        else:
            self.order = Order.from_url(self.client, self.order_uri)
            self.order.load_authorizations(self.client)
            self.authorizations.update(self.order.authorizations)
        authzs_to_wait_for = []
        for type_identifier, authz in self.authorizations.items():
            if authz.status == 'pending':
                if self.challenge is not None:
                    authz.call_validate(self.client, self.challenge, wait=False)
                    authzs_to_wait_for.append(authz)
                elif authz.status != 'valid':
                    authz.raise_error('Status is not "valid", even though no challenge should be necessary', module=self.client.module)
                self.changed = True
        wait_for_validation(authzs_to_wait_for, self.client)

    def download_alternate_chains(self, cert):
        alternate_chains = []
        for alternate in cert.alternates:
            try:
                alt_cert = CertificateChain.download(self.client, alternate)
            except ModuleFailException as e:
                self.module.warn('Error while downloading alternative certificate {0}: {1}'.format(alternate, e))
                continue
            alternate_chains.append(alt_cert)
        return alternate_chains

    def find_matching_chain(self, chains):
        for criterium_idx, matcher in enumerate(self.select_chain_matcher):
            for chain in chains:
                if matcher.match(chain):
                    self.module.debug('Found matching chain for criterium {0}'.format(criterium_idx))
                    return chain
        return None

    def get_certificate(self):
        """
        Request a new certificate and write it to the destination file.
        First verifies whether all authorizations are valid; if not, aborts
        with an error.
        """
        for identifier_type, identifier in self.identifiers:
            authz = self.authorizations.get(combine_identifier(identifier_type, identifier))
            if authz is None:
                raise ModuleFailException('Found no authorization information for "{identifier}"!'.format(identifier=combine_identifier(identifier_type, identifier)))
            if authz.status != 'valid':
                authz.raise_error('Status is "{status}" and not "valid"'.format(status=authz.status), module=self.module)
        if self.version == 1:
            cert = retrieve_acme_v1_certificate(self.client, pem_to_der(self.csr, self.csr_content))
        else:
            self.order.finalize(self.client, pem_to_der(self.csr, self.csr_content))
            cert = CertificateChain.download(self.client, self.order.certificate_uri)
            if self.module.params['retrieve_all_alternates'] or self.select_chain_matcher:
                alternate_chains = self.download_alternate_chains(cert)
                if self.module.params['retrieve_all_alternates']:
                    self.all_chains = [cert.to_json()]
                    for alt_chain in alternate_chains:
                        self.all_chains.append(alt_chain.to_json())
                if self.select_chain_matcher:
                    matching_chain = self.find_matching_chain([cert] + alternate_chains)
                    if matching_chain:
                        cert = matching_chain
                    else:
                        self.module.debug('Found no matching alternative chain')
        if cert.cert is not None:
            pem_cert = cert.cert
            chain = cert.chain
            if self.dest and write_file(self.module, self.dest, pem_cert.encode('utf8')):
                self.cert_days = self.client.backend.get_cert_days(self.dest)
                self.changed = True
            if self.fullchain_dest and write_file(self.module, self.fullchain_dest, (pem_cert + '\n'.join(chain)).encode('utf8')):
                self.cert_days = self.client.backend.get_cert_days(self.fullchain_dest)
                self.changed = True
            if self.chain_dest and write_file(self.module, self.chain_dest, '\n'.join(chain).encode('utf8')):
                self.changed = True

    def deactivate_authzs(self):
        """
        Deactivates all valid authz's. Does not raise exceptions.
        https://community.letsencrypt.org/t/authorization-deactivation/19860/2
        https://tools.ietf.org/html/rfc8555#section-7.5.2
        """
        for authz in self.authorizations.values():
            try:
                authz.deactivate(self.client)
            except Exception:
                pass
            if authz.status != 'deactivated':
                self.module.warn(warning='Could not deactivate authz object {0}.'.format(authz.url))