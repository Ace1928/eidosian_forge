import inspect
import sys
class KafkaStorageError(BrokerResponseError):
    errno = 56
    message = 'KAFKA_STORAGE_ERROR'
    description = 'The user-specified log directory is not found in the broker config.'