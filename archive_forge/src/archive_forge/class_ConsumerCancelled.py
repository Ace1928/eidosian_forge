from struct import pack, unpack
class ConsumerCancelled(RecoverableConnectionError):
    """AMQP Consumer Cancelled Predicate."""