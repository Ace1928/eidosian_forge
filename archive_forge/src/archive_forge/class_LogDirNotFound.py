import inspect
import sys
class LogDirNotFound(BrokerResponseError):
    errno = 57
    message = 'LOG_DIR_NOT_FOUND'
    description = 'The user-specified log directory is not found in the broker config.'