import abc
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class CertificateOrder(Order, CertificateOrderFormatter):
    _type = 'certificate'

    def __init__(self, api, name=None, status=None, created=None, updated=None, order_ref=None, container_ref=None, error_status_code=None, error_reason=None, sub_status=None, sub_status_message=None, creator_id=None, request_type=None, subject_dn=None, source_container_ref=None, ca_id=None, profile=None, request_data=None, requestor_name=None, requestor_email=None, requestor_phone=None):
        super(CertificateOrder, self).__init__(api, self._type, status=status, created=created, updated=updated, meta={'name': name, 'request_type': request_type, 'subject_dn': subject_dn, 'container_ref': source_container_ref, 'ca_id': ca_id, 'profile': profile, 'request_data': request_data, 'requestor_name': requestor_name, 'requestor_email': requestor_email, 'requestor_phone': requestor_phone}, order_ref=order_ref, error_status_code=error_status_code, error_reason=error_reason)
        self._container_ref = container_ref

    @property
    def container_ref(self):
        return self._container_ref

    def __repr__(self):
        return 'CertificateOrder(order_ref={0})'.format(self.order_ref)