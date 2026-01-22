from botocore.compat import six
from s3transfer.compat import accepts_kwargs
from s3transfer.exceptions import InvalidSubscriberMethodError
@classmethod
def _validate_subscriber_methods(cls):
    for subscriber_type in cls.VALID_SUBSCRIBER_TYPES:
        subscriber_method = getattr(cls, 'on_' + subscriber_type)
        if not six.callable(subscriber_method):
            raise InvalidSubscriberMethodError('Subscriber method %s must be callable.' % subscriber_method)
        if not accepts_kwargs(subscriber_method):
            raise InvalidSubscriberMethodError('Subscriber method %s must accept keyword arguments (**kwargs)' % subscriber_method)