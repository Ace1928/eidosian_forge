import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
class bfd(packet_base.PacketBase):
    """BFD (RFC 5880) Control packet encoder/decoder class.

    The serialized packet would looks like the ones described
    in the following sections.

    * RFC 5880 Generic BFD Control Packet Format

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.

    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============================== ============================================
    Attribute                      Description
    ============================== ============================================
    ver                            The version number of the protocol.
                                   This class implements protocol version 1.
    diag                           A diagnostic code specifying the local
                                   system's reason for the last change in
                                   session state.
    state                          The current BFD session state as seen by
                                   the transmitting system.
    flags                          Bitmap of the following flags:
                                   ``BFD_FLAG_POLL``,
                                   ``BFD_FLAG_FINAL``,
                                   ``BFD_FLAG_CTRL_PLANE_INDEP``,
                                   ``BFD_FLAG_AUTH_PRESENT``,
                                   ``BFD_FLAG_DEMAND``,
                                   ``BFD_FLAG_MULTIPOINT``
    detect_mult                    Detection time multiplier.
    my_discr                       My Discriminator.
    your_discr                     Your Discriminator.
    desired_min_tx_interval        Desired Min TX Interval. (in microseconds)
    required_min_rx_interval       Required Min RX Interval. (in microseconds)
    required_min_echo_rx_interval  Required Min Echo RX Interval.
                                   (in microseconds)
    auth_cls                       (Optional) Authentication Section instance.
                                   It's defined only when the Authentication
                                   Present (A) bit is set in flags.
                                   Assign an instance of the following classes:
                                   ``SimplePassword``, ``KeyedMD5``,
                                   ``MeticulousKeyedMD5``, ``KeyedSHA1``, and
                                   ``MeticulousKeyedSHA1``.
    length                         (Optional) Length of the BFD Control packet,
                                   in bytes.
    ============================== ============================================
    """
    _PACK_STR = '!BBBBIIIII'
    _PACK_STR_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': []}
    _auth_parsers = {}

    def __init__(self, ver=1, diag=0, state=0, flags=0, detect_mult=0, my_discr=0, your_discr=0, desired_min_tx_interval=0, required_min_rx_interval=0, required_min_echo_rx_interval=0, auth_cls=None, length=None):
        super(bfd, self).__init__()
        self.ver = ver
        self.diag = diag
        self.state = state
        self.flags = flags
        self.detect_mult = detect_mult
        self.my_discr = my_discr
        self.your_discr = your_discr
        self.desired_min_tx_interval = desired_min_tx_interval
        self.required_min_rx_interval = required_min_rx_interval
        self.required_min_echo_rx_interval = required_min_echo_rx_interval
        self.auth_cls = auth_cls
        if isinstance(length, int):
            self.length = length
        else:
            self.length = len(self)

    def __len__(self):
        if self.flags & BFD_FLAG_AUTH_PRESENT and self.auth_cls is not None:
            return self._PACK_STR_LEN + len(self.auth_cls)
        else:
            return self._PACK_STR_LEN

    @classmethod
    def parser(cls, buf):
        diag, flags, detect_mult, length, my_discr, your_discr, desired_min_tx_interval, required_min_rx_interval, required_min_echo_rx_interval = struct.unpack_from(cls._PACK_STR, buf[:cls._PACK_STR_LEN])
        ver = diag >> 5
        diag = diag & 31
        state = flags >> 6
        flags = flags & 63
        if flags & BFD_FLAG_AUTH_PRESENT:
            auth_type = operator.getitem(buf, cls._PACK_STR_LEN)
            auth_cls = cls._auth_parsers[auth_type].parser(buf[cls._PACK_STR_LEN:])[0]
        else:
            auth_cls = None
        msg = cls(ver, diag, state, flags, detect_mult, my_discr, your_discr, desired_min_tx_interval, required_min_rx_interval, required_min_echo_rx_interval, auth_cls)
        return (msg, None, None)

    def serialize(self, payload, prev):
        if self.flags & BFD_FLAG_AUTH_PRESENT and self.auth_cls is not None:
            return self.pack() + self.auth_cls.serialize(payload=None, prev=self)
        else:
            return self.pack()

    def pack(self):
        """
        Encode a BFD Control packet without authentication section.
        """
        diag = (self.ver << 5) + self.diag
        flags = (self.state << 6) + self.flags
        length = len(self)
        return struct.pack(self._PACK_STR, diag, flags, self.detect_mult, length, self.my_discr, self.your_discr, self.desired_min_tx_interval, self.required_min_rx_interval, self.required_min_echo_rx_interval)

    def authenticate(self, *args, **kwargs):
        """Authenticate this packet.

        Returns a boolean indicates whether the packet can be authenticated
        or not.

        Returns ``False`` if the Authentication Present (A) is not set in the
        flag of this packet.

        Returns ``False`` if the Authentication Section for this packet is not
        present.

        For the description of the arguemnts of this method, refer to the
        authentication method of the Authentication Section classes.
        """
        if not self.flags & BFD_FLAG_AUTH_PRESENT or not issubclass(self.auth_cls.__class__, BFDAuth):
            return False
        return self.auth_cls.authenticate(self, *args, **kwargs)

    @classmethod
    def set_auth_parser(cls, auth_cls):
        cls._auth_parsers[auth_cls.auth_type] = auth_cls

    @classmethod
    def register_auth_type(cls, auth_type):

        def _set_type(auth_cls):
            auth_cls.set_type(auth_cls, auth_type)
            cls.set_auth_parser(auth_cls)
            return auth_cls
        return _set_type