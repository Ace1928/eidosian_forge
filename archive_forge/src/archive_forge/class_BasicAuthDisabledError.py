from __future__ import unicode_literals
class BasicAuthDisabledError(AuthenticationError):
    message = 'WinRM/HTTP Basic authentication is not enabled on remote host'