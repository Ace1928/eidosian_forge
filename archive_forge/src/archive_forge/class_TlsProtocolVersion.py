import enum
import typing
class TlsProtocolVersion(enum.IntEnum):
    tls1_0 = 769
    tls1_1 = 770
    tls1_2 = 771
    tls1_3 = 772

    @classmethod
    def native_labels(cls) -> typing.Dict['TlsProtocolVersion', str]:
        return {TlsProtocolVersion.tls1_0: 'TLS 1.0 (0x0301)', TlsProtocolVersion.tls1_1: 'TLS 1.1 (0x0302)', TlsProtocolVersion.tls1_2: 'TLS 1.2 (0x0303)', TlsProtocolVersion.tls1_3: 'TLS 1.3 (0x0304)'}

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown TLS Protocol Version 0x{0:04X}')