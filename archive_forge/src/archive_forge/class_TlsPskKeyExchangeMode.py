import enum
import typing
class TlsPskKeyExchangeMode(enum.IntEnum):
    psk_ke = 0
    psk_dhe_ke = 1

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown PSK Key Exchange Mode 0x{0:02X}')