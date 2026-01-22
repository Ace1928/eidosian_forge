import warnings
from twisted.trial.unittest import TestCase
class TRANSPORT_MESSAGE(Values):
    """
            Message types supported by an SSH transport.
            """
    KEX_DH_GEX_REQUEST_OLD = ValueConstant(30)
    KEXDH_INIT = ValueConstant(30)