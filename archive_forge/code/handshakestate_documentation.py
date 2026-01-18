from dissononce.processing.symmetricstate import SymmetricState
from dissononce.processing.handshakestate import HandshakeState as BaseHandshakeState
from dissononce.dh.public import PublicKey
from dissononce.dh.keypair import KeyPair
from dissononce.dh.dh import DH
import logging

        :param message:
        :type message: bytes
        :param payload_buffer:
        :type payload_buffer: bytearray
        :return:
        :rtype: tuple[dissononce.processing.cipherstate.CipherState] | None
        