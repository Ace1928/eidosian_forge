from tempest.lib import exceptions
class ShareReplicaBuildErrorException(exceptions.TempestException):
    message = 'Share replica %(replica)s failed to build and is in ERROR status.'