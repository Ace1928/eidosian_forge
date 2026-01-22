from dissononce.dh.x25519.x25519 import PublicKey
from ..proto import wa20_pb2
import axolotl_curve25519 as curve
import logging
import time

        :param rs:
        :type rs: PublicKey
        :param certificate_data:
        :type certificate_data: bytes
        :return:
        :rtype:
        