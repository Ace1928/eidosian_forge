import struct
import os
import sys
from paramiko.common import MSG_USERAUTH_REQUEST
from paramiko.ssh_exception import SSHException
from paramiko._version import __version_info__
class _SSH_GSSAuth:
    """
    Contains the shared variables and methods of `._SSH_GSSAPI_OLD`,
    `._SSH_GSSAPI_NEW` and `._SSH_SSPI`.
    """

    def __init__(self, auth_method, gss_deleg_creds):
        """
        :param str auth_method: The name of the SSH authentication mechanism
                                (gssapi-with-mic or gss-keyex)
        :param bool gss_deleg_creds: Delegate client credentials or not
        """
        self._auth_method = auth_method
        self._gss_deleg_creds = gss_deleg_creds
        self._gss_host = None
        self._username = None
        self._session_id = None
        self._service = 'ssh-connection'
        '\n        OpenSSH supports Kerberos V5 mechanism only for GSS-API authentication,\n        so we also support the krb5 mechanism only.\n        '
        self._krb5_mech = '1.2.840.113554.1.2.2'
        self._gss_ctxt = None
        self._gss_ctxt_status = False
        self._gss_srv_ctxt = None
        self._gss_srv_ctxt_status = False
        self.cc_file = None

    def set_service(self, service):
        """
        This is just a setter to use a non default service.
        I added this method, because RFC 4462 doesn't specify "ssh-connection"
        as the only service value.

        :param str service: The desired SSH service
        """
        if service.find('ssh-'):
            self._service = service

    def set_username(self, username):
        """
        Setter for C{username}. If GSS-API Key Exchange is performed, the
        username is not set by C{ssh_init_sec_context}.

        :param str username: The name of the user who attempts to login
        """
        self._username = username

    def ssh_gss_oids(self, mode='client'):
        """
        This method returns a single OID, because we only support the
        Kerberos V5 mechanism.

        :param str mode: Client for client mode and server for server mode
        :return: A byte sequence containing the number of supported
                 OIDs, the length of the OID and the actual OID encoded with
                 DER
        :note: In server mode we just return the OID length and the DER encoded
               OID.
        """
        from pyasn1.type.univ import ObjectIdentifier
        from pyasn1.codec.der import encoder
        OIDs = self._make_uint32(1)
        krb5_OID = encoder.encode(ObjectIdentifier(self._krb5_mech))
        OID_len = self._make_uint32(len(krb5_OID))
        if mode == 'server':
            return OID_len + krb5_OID
        return OIDs + OID_len + krb5_OID

    def ssh_check_mech(self, desired_mech):
        """
        Check if the given OID is the Kerberos V5 OID (server mode).

        :param str desired_mech: The desired GSS-API mechanism of the client
        :return: ``True`` if the given OID is supported, otherwise C{False}
        """
        from pyasn1.codec.der import decoder
        mech, __ = decoder.decode(desired_mech)
        if mech.__str__() != self._krb5_mech:
            return False
        return True

    def _make_uint32(self, integer):
        """
        Create a 32 bit unsigned integer (The byte sequence of an integer).

        :param int integer: The integer value to convert
        :return: The byte sequence of an 32 bit integer
        """
        return struct.pack('!I', integer)

    def _ssh_build_mic(self, session_id, username, service, auth_method):
        """
        Create the SSH2 MIC filed for gssapi-with-mic.

        :param str session_id: The SSH session ID
        :param str username: The name of the user who attempts to login
        :param str service: The requested SSH service
        :param str auth_method: The requested SSH authentication mechanism
        :return: The MIC as defined in RFC 4462. The contents of the
                 MIC field are:
                 string    session_identifier,
                 byte      SSH_MSG_USERAUTH_REQUEST,
                 string    user-name,
                 string    service (ssh-connection),
                 string    authentication-method
                           (gssapi-with-mic or gssapi-keyex)
        """
        mic = self._make_uint32(len(session_id))
        mic += session_id
        mic += struct.pack('B', MSG_USERAUTH_REQUEST)
        mic += self._make_uint32(len(username))
        mic += username.encode()
        mic += self._make_uint32(len(service))
        mic += service.encode()
        mic += self._make_uint32(len(auth_method))
        mic += auth_method.encode()
        return mic