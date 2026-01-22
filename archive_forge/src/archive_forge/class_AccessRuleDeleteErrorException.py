from tempest.lib import exceptions
class AccessRuleDeleteErrorException(exceptions.TempestException):
    message = 'Access rule %(access)s failed to delete and is in ERROR state.'