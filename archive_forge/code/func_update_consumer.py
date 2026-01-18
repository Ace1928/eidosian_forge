import abc
import string
from keystone import exception
@abc.abstractmethod
def update_consumer(self, consumer_id, consumer_ref):
    """Update consumer.

        :param consumer_id: id of consumer to update
        :type consumer_id: string
        :param consumer_ref: new consumer ref with consumer name
        :type consumer_ref: dict
        :returns: consumer_ref

        """
    raise exception.NotImplemented()