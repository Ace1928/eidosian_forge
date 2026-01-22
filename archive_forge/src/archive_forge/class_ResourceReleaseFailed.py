from tempest.lib import exceptions
class ResourceReleaseFailed(exceptions.TempestException):
    message = "Failed to release resource '%(res_type)s' with id '%(res_id)s'."