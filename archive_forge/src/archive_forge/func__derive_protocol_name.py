from dissononce.processing.symmetricstate import SymmetricState
from dissononce.processing.handshakestate import HandshakeState as BaseHandshakeState
from dissononce.dh.public import PublicKey
from dissononce.dh.keypair import KeyPair
from dissononce.dh.dh import DH
import logging
def _derive_protocol_name(self, handshake_pattern_name):
    return self.__class__._TEMPLATE_PROTOCOL_NAME.format(handshake=handshake_pattern_name, dh=self._dh.name, cipher=self._symmetricstate.ciphername, hash=self._symmetricstate.hashname)