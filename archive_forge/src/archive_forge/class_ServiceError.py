from keystonemiddleware import exceptions
class ServiceError(exceptions.KeystoneMiddlewareException):
    pass