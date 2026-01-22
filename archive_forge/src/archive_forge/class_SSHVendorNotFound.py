class SSHVendorNotFound(BzrError):
    _fmt = "Don't know how to handle SSH connections. Please set BRZ_SSH environment variable."