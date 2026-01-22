import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class DomainScopedPayload(BasePayload):
    version = 1

    @classmethod
    def assemble(cls, user_id, methods, system, project_id, domain_id, expires_at, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint):
        b_user_id = cls.attempt_convert_uuid_hex_to_bytes(user_id)
        methods = auth_plugins.convert_method_list_to_integer(methods)
        try:
            b_domain_id = cls.convert_uuid_hex_to_bytes(domain_id)
        except ValueError:
            if domain_id == CONF.identity.default_domain_id:
                b_domain_id = domain_id
            else:
                raise
        expires_at_int = cls._convert_time_string_to_float(expires_at)
        b_audit_ids = list(map(cls.random_urlsafe_str_to_bytes, audit_ids))
        return (b_user_id, methods, b_domain_id, expires_at_int, b_audit_ids)

    @classmethod
    def disassemble(cls, payload):
        is_stored_as_bytes, user_id = payload[0]
        user_id = cls._convert_or_decode(is_stored_as_bytes, user_id)
        methods = auth_plugins.convert_integer_to_method_list(payload[1])
        try:
            domain_id = cls.convert_uuid_bytes_to_hex(payload[2])
        except ValueError:
            if isinstance(payload[2], bytes):
                payload[2] = payload[2].decode('utf-8')
            if payload[2] == CONF.identity.default_domain_id:
                domain_id = payload[2]
            else:
                raise
        expires_at_str = cls._convert_float_to_time_string(payload[3])
        audit_ids = list(map(cls.base64_encode, payload[4]))
        system = None
        project_id = None
        trust_id = None
        federated_group_ids = None
        identity_provider_id = None
        protocol_id = None
        access_token_id = None
        app_cred_id = None
        thumbprint = None
        return (user_id, methods, system, project_id, domain_id, expires_at_str, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint)