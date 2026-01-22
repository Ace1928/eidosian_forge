import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
@vrrp.register_vrrp_version(VRRP_VERSION_V2, 1)
class vrrpv2(vrrp):
    """VRRPv2 (RFC 3768) header encoder/decoder class.

    Unlike other os_ken.lib.packet.packet_base.PacketBase derived classes,
    *create* method should be used to instantiate an object of this class.
    """
    _PACK_STR = '!BBBBBBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _CHECKSUM_PACK_STR = '!H'
    _CHECKSUM_OFFSET = 6
    _AUTH_DATA_PACK_STR = '!II'
    _AUTH_DATA_LEN = struct.calcsize('!II')

    def __len__(self):
        return self._MIN_LEN + self._IPV4_ADDRESS_LEN * self.count_ip + self._AUTH_DATA_LEN

    def checksum_ok(self, ipvx, vrrp_buf):
        return packet_utils.checksum(vrrp_buf) == 0

    @staticmethod
    def create(type_, vrid, priority, max_adver_int, ip_addresses):
        """Unlike other os_ken.lib.packet.packet_base.PacketBase derived classes,
        this method should be used to instantiate an object of this class.

        This method's arguments are same as os_ken.lib.packet.vrrp.vrrp object's
        attributes of the same name.  (except that *type_* corresponds to
        *type* attribute.)
        """
        return vrrp.create_version(VRRP_VERSION_V2, type_, vrid, priority, max_adver_int, ip_addresses, auth_type=VRRP_AUTH_NO_AUTH, auth_data=VRRP_AUTH_DATA)

    @staticmethod
    def _ip_addresses_pack_str(count_ip):
        return '!' + vrrpv2._IPV4_ADDRESS_PACK_STR_RAW * count_ip

    @classmethod
    def parser(cls, buf):
        version_type, vrid, priority, count_ip, auth_type, adver_int, checksum = struct.unpack_from(cls._PACK_STR, buf)
        version, type_ = vrrp_from_version_type(version_type)
        offset = cls._MIN_LEN
        ip_addresses_pack_str = cls._ip_addresses_pack_str(count_ip)
        ip_addresses_bin = struct.unpack_from(ip_addresses_pack_str, buf, offset)
        ip_addresses = [addrconv.ipv4.bin_to_text(x) for x in ip_addresses_bin]
        offset += struct.calcsize(ip_addresses_pack_str)
        auth_data = struct.unpack_from(cls._AUTH_DATA_PACK_STR, buf, offset)
        msg = cls(version, type_, vrid, priority, count_ip, adver_int, checksum, ip_addresses, auth_type, auth_data)
        return (msg, None, buf[len(msg):])

    @staticmethod
    def serialize_static(vrrp_, prev):
        assert not vrrp_.is_ipv6
        ip_addresses_pack_str = vrrpv2._ip_addresses_pack_str(vrrp_.count_ip)
        ip_addresses_len = struct.calcsize(ip_addresses_pack_str)
        vrrp_len = vrrpv2._MIN_LEN + ip_addresses_len + vrrpv2._AUTH_DATA_LEN
        checksum = False
        if vrrp_.checksum is None:
            checksum = True
            vrrp_.checksum = 0
        if vrrp_.auth_type is None:
            vrrp_.auth_type = VRRP_AUTH_NO_AUTH
        if vrrp_.auth_data is None:
            vrrp_.auth_data = VRRP_AUTH_DATA
        buf = bytearray(vrrp_len)
        offset = 0
        struct.pack_into(vrrpv2._PACK_STR, buf, offset, vrrp_to_version_type(vrrp_.version, vrrp_.type), vrrp_.vrid, vrrp_.priority, vrrp_.count_ip, vrrp_.auth_type, vrrp_.max_adver_int, vrrp_.checksum)
        offset += vrrpv2._MIN_LEN
        struct.pack_into(ip_addresses_pack_str, buf, offset, *[addrconv.ipv4.text_to_bin(x) for x in vrrp_.ip_addresses])
        offset += ip_addresses_len
        struct.pack_into(vrrpv2._AUTH_DATA_PACK_STR, buf, offset, *vrrp_.auth_data)
        if checksum:
            vrrp_.checksum = packet_utils.checksum(buf)
            struct.pack_into(vrrpv2._CHECKSUM_PACK_STR, buf, vrrpv2._CHECKSUM_OFFSET, vrrp_.checksum)
        return buf

    def is_valid(self):
        return self.version == VRRP_VERSION_V2 and self.type == VRRP_TYPE_ADVERTISEMENT and (VRRP_VRID_MIN <= self.vrid) and (self.vrid <= VRRP_VRID_MAX) and (VRRP_PRIORITY_MIN <= self.priority) and (self.priority <= VRRP_PRIORITY_MAX) and (self.auth_type == VRRP_AUTH_NO_AUTH) and (VRRP_V2_MAX_ADVER_INT_MIN <= self.max_adver_int) and (self.max_adver_int <= VRRP_V2_MAX_ADVER_INT_MAX) and (self.count_ip == len(self.ip_addresses))