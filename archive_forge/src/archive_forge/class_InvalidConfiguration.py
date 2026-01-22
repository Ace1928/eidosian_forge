from tempest.lib import exceptions
class InvalidConfiguration(exceptions.TempestException):
    message = 'Invalid configuration: %(reason)s'