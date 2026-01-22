from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IpConfiguration(_messages.Message):
    """IP Management configuration.

  Enums:
    SslModeValueValuesEnum: Specify how SSL/TLS is enforced in database
      connections. If you must use the `require_ssl` flag for backward
      compatibility, then only the following value pairs are valid: For
      PostgreSQL and MySQL: * `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and
      `require_ssl=false` * `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false`
      * `ssl_mode=TRUSTED_CLIENT_CERTIFICATE_REQUIRED` and `require_ssl=true`
      For SQL Server: * `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and
      `require_ssl=false` * `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=true`
      The value of `ssl_mode` gets priority over the value of `require_ssl`.
      For example, for the pair `ssl_mode=ENCRYPTED_ONLY` and
      `require_ssl=false`, the `ssl_mode=ENCRYPTED_ONLY` means only accept SSL
      connections, while the `require_ssl=false` means accept both non-SSL and
      SSL connections. MySQL and PostgreSQL databases respect `ssl_mode` in
      this case and accept only SSL connections.

  Fields:
    allocatedIpRange: The name of the allocated ip range for the private ip
      Cloud SQL instance. For example: "google-managed-services-default". If
      set, the instance ip will be created in the allocated range. The range
      name must comply with [RFC 1035](https://tools.ietf.org/html/rfc1035).
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?.`
    authorizedNetworks: The list of external networks that are allowed to
      connect to the instance using the IP. In 'CIDR' notation, also known as
      'slash' notation (for example: `157.197.200.0/24`).
    enablePrivatePathForGoogleCloudServices: Controls connectivity to private
      IP instances from Google services, such as BigQuery.
    ipv4Enabled: Whether the instance is assigned a public IP address or not.
    privateNetwork: The resource link for the VPC network from which the Cloud
      SQL instance is accessible for private IP. For example,
      `/projects/myProject/global/networks/default`. This setting can be
      updated, but it cannot be removed after it is set.
    pscConfig: PSC settings for this instance.
    requireSsl: Use `ssl_mode` instead. Whether SSL/TLS connections over IP
      are enforced. If set to false, then allow both non-SSL/non-TLS and
      SSL/TLS connections. For SSL/TLS connections, the client certificate
      won't be verified. If set to true, then only allow connections encrypted
      with SSL/TLS and with valid client certificates. If you want to enforce
      SSL/TLS without enforcing the requirement for valid client certificates,
      then use the `ssl_mode` flag instead of the legacy `require_ssl` flag.
    reservedIpRange: This field is deprecated and will be removed from a
      future version of the API.
    sslMode: Specify how SSL/TLS is enforced in database connections. If you
      must use the `require_ssl` flag for backward compatibility, then only
      the following value pairs are valid: For PostgreSQL and MySQL: *
      `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and `require_ssl=false` *
      `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false` *
      `ssl_mode=TRUSTED_CLIENT_CERTIFICATE_REQUIRED` and `require_ssl=true`
      For SQL Server: * `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and
      `require_ssl=false` * `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=true`
      The value of `ssl_mode` gets priority over the value of `require_ssl`.
      For example, for the pair `ssl_mode=ENCRYPTED_ONLY` and
      `require_ssl=false`, the `ssl_mode=ENCRYPTED_ONLY` means only accept SSL
      connections, while the `require_ssl=false` means accept both non-SSL and
      SSL connections. MySQL and PostgreSQL databases respect `ssl_mode` in
      this case and accept only SSL connections.
  """

    class SslModeValueValuesEnum(_messages.Enum):
        """Specify how SSL/TLS is enforced in database connections. If you must
    use the `require_ssl` flag for backward compatibility, then only the
    following value pairs are valid: For PostgreSQL and MySQL: *
    `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and `require_ssl=false` *
    `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false` *
    `ssl_mode=TRUSTED_CLIENT_CERTIFICATE_REQUIRED` and `require_ssl=true` For
    SQL Server: * `ssl_mode=ALLOW_UNENCRYPTED_AND_ENCRYPTED` and
    `require_ssl=false` * `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=true` The
    value of `ssl_mode` gets priority over the value of `require_ssl`. For
    example, for the pair `ssl_mode=ENCRYPTED_ONLY` and `require_ssl=false`,
    the `ssl_mode=ENCRYPTED_ONLY` means only accept SSL connections, while the
    `require_ssl=false` means accept both non-SSL and SSL connections. MySQL
    and PostgreSQL databases respect `ssl_mode` in this case and accept only
    SSL connections.

    Values:
      SSL_MODE_UNSPECIFIED: The SSL mode is unknown.
      ALLOW_UNENCRYPTED_AND_ENCRYPTED: Allow non-SSL/non-TLS and SSL/TLS
        connections. For SSL/TLS connections, the client certificate won't be
        verified. When this value is used, the legacy `require_ssl` flag must
        be false or cleared to avoid the conflict between values of two flags.
      ENCRYPTED_ONLY: Only allow connections encrypted with SSL/TLS. When this
        value is used, the legacy `require_ssl` flag must be false or cleared
        to avoid the conflict between values of two flags.
      TRUSTED_CLIENT_CERTIFICATE_REQUIRED: Only allow connections encrypted
        with SSL/TLS and with valid client certificates. When this value is
        used, the legacy `require_ssl` flag must be true or cleared to avoid
        the conflict between values of two flags. PostgreSQL clients or users
        that connect using IAM database authentication must use either the
        [Cloud SQL Auth
        Proxy](https://cloud.google.com/sql/docs/postgres/connect-auth-proxy)
        or [Cloud SQL
        Connectors](https://cloud.google.com/sql/docs/postgres/connect-
        connectors) to enforce client identity verification. This value is not
        applicable to SQL Server.
    """
        SSL_MODE_UNSPECIFIED = 0
        ALLOW_UNENCRYPTED_AND_ENCRYPTED = 1
        ENCRYPTED_ONLY = 2
        TRUSTED_CLIENT_CERTIFICATE_REQUIRED = 3
    allocatedIpRange = _messages.StringField(1)
    authorizedNetworks = _messages.MessageField('AclEntry', 2, repeated=True)
    enablePrivatePathForGoogleCloudServices = _messages.BooleanField(3)
    ipv4Enabled = _messages.BooleanField(4)
    privateNetwork = _messages.StringField(5)
    pscConfig = _messages.MessageField('PscConfig', 6)
    requireSsl = _messages.BooleanField(7)
    reservedIpRange = _messages.StringField(8)
    sslMode = _messages.EnumField('SslModeValueValuesEnum', 9)