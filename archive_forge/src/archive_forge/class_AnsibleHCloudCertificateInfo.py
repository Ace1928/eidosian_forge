from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.certificates import BoundCertificate
class AnsibleHCloudCertificateInfo(AnsibleHCloud):
    represent = 'hcloud_certificate_info'
    hcloud_certificate_info: list[BoundCertificate] | None = None

    def _prepare_result(self):
        certificates = []
        for certificate in self.hcloud_certificate_info:
            if certificate:
                certificates.append({'id': to_native(certificate.id), 'name': to_native(certificate.name), 'fingerprint': to_native(certificate.fingerprint), 'certificate': to_native(certificate.certificate), 'not_valid_before': to_native(certificate.not_valid_before), 'not_valid_after': to_native(certificate.not_valid_after), 'domain_names': [to_native(domain) for domain in certificate.domain_names], 'labels': certificate.labels})
        return certificates

    def get_certificates(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_certificate_info = [self.client.certificates.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_certificate_info = [self.client.certificates.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_certificate_info = self.client.certificates.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_certificate_info = self.client.certificates.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)