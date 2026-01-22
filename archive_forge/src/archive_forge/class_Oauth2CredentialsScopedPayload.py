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
class Oauth2CredentialsScopedPayload(BasePayload):
    version = 10

    @classmethod
    def assemble(cls, user_id, methods, system, project_id, domain_id, expires_at, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint):
        b_user_id = cls.attempt_convert_uuid_hex_to_bytes(user_id)
        methods = auth_plugins.convert_method_list_to_integer(methods)
        b_project_id = cls.attempt_convert_uuid_hex_to_bytes(project_id)
        b_domain_id = cls.attempt_convert_uuid_hex_to_bytes(domain_id)
        expires_at_int = cls._convert_time_string_to_float(expires_at)
        b_audit_ids = list(map(cls.random_urlsafe_str_to_bytes, audit_ids))
        b_thumbprint = (False, thumbprint)
        return (b_user_id, methods, b_project_id, b_domain_id, expires_at_int, b_audit_ids, b_thumbprint)

    @classmethod
    def disassemble(cls, payload):
        is_stored_as_bytes, user_id = payload[0]
        user_id = cls._convert_or_decode(is_stored_as_bytes, user_id)
        methods = auth_plugins.convert_integer_to_method_list(payload[1])
        is_stored_as_bytes, project_id = payload[2]
        project_id = cls._convert_or_decode(is_stored_as_bytes, project_id)
        is_stored_as_bytes, domain_id = payload[3]
        domain_id = cls._convert_or_decode(is_stored_as_bytes, domain_id)
        expires_at_str = cls._convert_float_to_time_string(payload[4])
        audit_ids = list(map(cls.base64_encode, payload[5]))
        is_stored_as_bytes, thumbprint = payload[6]
        thumbprint = cls._convert_or_decode(is_stored_as_bytes, thumbprint)
        system = None
        trust_id = None
        federated_group_ids = None
        identity_provider_id = None
        protocol_id = None
        access_token_id = None
        app_cred_id = None
        return (user_id, methods, system, project_id, domain_id, expires_at_str, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint)