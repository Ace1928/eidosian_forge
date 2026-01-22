import enum
import typing
class TlsSignatureScheme(enum.IntEnum):
    rsa_pkcs1_sha1 = 513
    dsa_sha1 = 514
    ecdsa_sha1 = 515
    sha224_rsa = 769
    dsa_sha224 = 770
    sha224_ecdsa = 771
    rsa_pkcs1_sha256 = 1025
    dsa_sha256 = 1026
    ecdsa_secp256r1_sha256 = 1027
    rsa_pkcs1_sha256_legacy = 1056
    rsa_pkcs1_sha384 = 1281
    dsa_sha384 = 1282
    ecdsa_secp384r1_sha384 = 1283
    rsa_pkcs1_sha384_legacy = 1312
    rsa_pkcs1_sha512 = 1537
    dsa_sha512 = 1538
    ecdsa_secp521r1_sha512 = 1539
    rsa_pkcs1_sha512_legacy = 1568
    eccsi_sha256 = 1796
    iso_ibs1 = 1797
    iso_ibs2 = 1798
    iso_chinese_ibs = 1799
    sm2sig_sm3 = 1800
    gostr34102012_256a = 1801
    gostr34102012_256b = 1802
    gostr34102012_256c = 1803
    gostr34102012_256d = 1804
    gostr34102012_512a = 1805
    gostr34102012_512b = 1806
    gostr34102012_512c = 1807
    rsa_pss_rsae_sha256 = 2052
    rsa_pss_rsae_sha384 = 2053
    rsa_pss_rsae_sha512 = 2054
    ed25519 = 2055
    ed448 = 2056
    rsa_pss_pss_sha256 = 2057
    rsa_pss_pss_sha384 = 2058
    rsa_pss_pss_sha512 = 2059
    ecdsa_brainpoolP256r1tls13_sha256 = 2074
    ecdsa_brainpoolP384r1tls13_sha384 = 2075
    ecdsa_brainpoolP512r1tls13_sha512 = 2076

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown Signature Scheme 0x{0:04X}')