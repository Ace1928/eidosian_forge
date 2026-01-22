from __future__ import unicode_literals
class WinRMTransportError(Exception):
    """WinRM errors specific to transport-level problems (unexpected HTTP error codes, etc)"""

    @property
    def protocol(self):
        return self.args[0]

    @property
    def code(self):
        return self.args[1]

    @property
    def message(self):
        return 'Bad HTTP response returned from server. Code {0}'.format(self.code)

    @property
    def response_text(self):
        return self.args[2]

    def __str__(self):
        return self.message