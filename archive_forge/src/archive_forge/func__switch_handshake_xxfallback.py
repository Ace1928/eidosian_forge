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
def _switch_handshake_xxfallback(self, stream, s, client_payload, server_hello):
    """
        :param handshake_pattern:
        :type handshake_pattern: HandshakePattern
        :param stream:
        :type stream: SegmentedStream
        :param s:
        :type s: KeyPair
        :param e:
        :type e: KeyPair
        :param client_payload:
        :type client_payload:
        :param server_hello:
        :type server_hello:
        :return:
        :rtype: tuple(CipherState,CipherState)
        """
    self._handshakestate.switch(handshake_pattern=FallbackPatternModifier().modify(XXHandshakePattern()), initiator=True, prologue=self._prologue, s=s)
    payload_buffer = bytearray()
    self._handshakestate.read_message(server_hello.ephemeral + server_hello.static + server_hello.payload, payload_buffer)
    certman = CertMan()
    if certman.is_valid(self._handshakestate.rs, bytes(payload_buffer)):
        logger.debug('cert is valid')
    else:
        logger.error('cert is not valid')
    message_buffer = bytearray()
    cipherpair = self._handshakestate.write_message(client_payload.SerializeToString(), message_buffer)
    static, payload = ByteUtil.split(bytes(message_buffer), 48, len(message_buffer) - 48)
    client_finish = wa20_pb2.HandshakeMessage.ClientFinish()
    client_finish.static = static
    client_finish.payload = payload
    outgoing_handshakemessage = wa20_pb2.HandshakeMessage()
    outgoing_handshakemessage.client_finish.MergeFrom(client_finish)
    stream.write_segment(outgoing_handshakemessage.SerializeToString())
    return cipherpair