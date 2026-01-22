class ServerUnreachable(IntegrationException):
    message = 'The server is not reachable via the configured network'