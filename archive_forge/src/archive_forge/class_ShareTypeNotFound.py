from tempest.lib import exceptions
class ShareTypeNotFound(exceptions.NotFound):
    message = "Share type '%(share_type)s' was not found."