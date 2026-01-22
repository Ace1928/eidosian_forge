import random
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.extras.processing.handshakestate_guarded import GuardedHandshakeState
from dissononce.extras.processing.handshakestate_switchable import SwitchableHandshakeState
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.handshakepatterns.interactive.IK import IKHandshakePattern
from dissononce.processing.handshakepatterns.interactive.XX import XXHandshakePattern
from dissononce.processing.modifiers.fallback import FallbackPatternModifier
from dissononce.processing.impl.cipherstate import CipherState
from dissononce.cipher.aesgcm import AESGCMCipher
from dissononce.hash.sha256 import SHA256Hash
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.private import PrivateKey
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.extras.dh.dangerous.dh_nogen import NoGenDH
from dissononce.exceptions.decrypt import DecryptFailedException
from google.protobuf.message import DecodeError
from .dissononce_extras.processing.symmetricstate_wa import WASymmetricState
from .proto import wa20_pb2
from .streams.segmented.segmented import SegmentedStream
from .certman.certman import CertMan
from .exceptions.new_rs_exception import NewRemoteStaticException
from .config.client import ClientConfig
from .structs.publickey import PublicKey
from .util.byte import ByteUtil
from.exceptions.handshake_failed_exception import HandshakeFailedException
import logging
def _create_full_payload(self, client_config):
    """
        :param client_config:
        :type client_config: ClientConfig
        :return:
        :rtype: wa20_pb2.ClientPayload
        """
    client_payload = wa20_pb2.ClientPayload()
    user_agent = wa20_pb2.ClientPayload.UserAgent()
    user_agent_app_version = wa20_pb2.ClientPayload.UserAgent.AppVersion()
    user_agent.platform = client_config.useragent.platform
    user_agent.mcc = client_config.useragent.mcc
    user_agent.mnc = client_config.useragent.mnc
    user_agent.os_version = client_config.useragent.os_version
    user_agent.manufacturer = client_config.useragent.manufacturer
    user_agent.device = client_config.useragent.device
    user_agent.os_build_number = client_config.useragent.os_build_number
    user_agent.phone_id = client_config.useragent.phone_id
    user_agent.locale_language_iso_639_1 = client_config.useragent.locale_lang
    user_agent.locale_country_iso_3166_1_alpha_2 = client_config.useragent.locale_country
    user_agent_app_version.primary = client_config.useragent.app_version.primary
    user_agent_app_version.secondary = client_config.useragent.app_version.secondary
    user_agent_app_version.tertiary = client_config.useragent.app_version.tertiary
    user_agent_app_version.quaternary = client_config.useragent.app_version.quaternary
    user_agent.app_version.MergeFrom(user_agent_app_version)
    client_payload.username = client_config.username
    client_payload.passive = client_config.passive
    client_payload.push_name = client_config.pushname
    max_int = 2 ** 32 / 2
    client_payload.session_id = random.randint(-max_int, max_int - 1)
    client_payload.short_connect = client_config.short_connect
    client_payload.connect_type = 1
    client_payload.user_agent.MergeFrom(user_agent)
    return client_payload