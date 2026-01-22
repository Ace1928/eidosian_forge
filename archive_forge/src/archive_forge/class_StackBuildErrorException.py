class StackBuildErrorException(IntegrationException):
    message = "Stack %(stack_identifier)s is in %(stack_status)s status due to '%(stack_status_reason)s'"