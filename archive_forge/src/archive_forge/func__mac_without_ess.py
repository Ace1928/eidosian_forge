import struct
import typing
from spnego._ntlm_raw.crypto import RC4Handle, crc32, hmac_md5, rc4
from spnego._ntlm_raw.messages import NegotiateFlags
from spnego.exceptions import OperationNotAvailableError
def _mac_without_ess(handle: RC4Handle, seq_num: int, b_data: bytes) -> bytes:
    """NTLM MAC without Extended Session Security

    Generates the NTLM signature when Extended Session Security has not been negotiated. The structure of the signature
    is documented at `NTLM signature without ESS`_.

    The algorithm as documented by `MAC without ESS`_ is::

        Define MAC(Handle, SigningKey, SeqNum, Message) as
            Set NTLMSSP_MESSAGE_SIGNATURE.Version to 0x00000001
            Set NTLMSSP_MESSAGE_SIGNATURE.Checksum to CRC32(Message)
            Set NTLMSSP_MESSAGE_SIGNATURE.RandomPad RC4(Handle, RandomPad)
            Set NTLMSSP_MESSAGE_SIGNATURE.Checksum to RC4(Handle, NTLMSSP_MESSAGE_SIGNATURE.Checksum)
            Set NTLMSSP_MESSAGE_SIGNATURE.SeqNum to RC4(Handle, 0x00000000)

            If (connection oriented)
                Set NTLMSSP_MESSAGE_SIGNATURE.SeqNum to NTLMSSP_MESSAGE_SIGNATURE.SeqNum XOR SeqNum
                Set SeqNum to SeqNum + 1

            Else
                Set NTLMSSP_MESSAGE_SIGNATURE.SeqNum to NTLMSSP_MESSAGE_SIGNATURE.SeqNum XOR (app supplied SeqNum)

            Endif

            Set NTLMSSP_MESSAGE_SIGNATURE.RandomPad to 0

        EndDefine

    Args:
        handle: The RC4 handle for the negotiated context.
        seq_num: The sequence number for the signature.
        b_data: The data/message bytes to sign.

    Returns:
        bytes: The NTLM signature.

    .. _NTLM signature without ESS:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/83fbd0e7-8ab0-4873-8cbe-795249b46b8a

    .. _MAC without ESS:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/0b1fb6a6-7224-4d5b-af35-fdd45c0791e5
    """
    checksum = crc32(b_data)
    rc4(handle, b'\x00\x00\x00\x00')
    checksum = rc4(handle, checksum)
    temp_seq_num = struct.unpack('<I', rc4(handle, b'\x00\x00\x00\x00'))[0]
    b_seq_num = struct.pack('<I', temp_seq_num ^ seq_num)
    return b'\x01\x00\x00\x00' + b'\x00\x00\x00\x00' + checksum + b_seq_num