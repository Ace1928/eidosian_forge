import logging
class ExtraLogFormatter(logging.Formatter):
    """
    Custom log formatter which attaches all the attributes from the "extra"
    dictionary which start with an underscore to the end of the log message.

    For example:
    extra={'_id': 'user-1', '_path': '/foo/bar'}
    """

    def format(self, record):
        custom_attributes = {k: v for k, v in record.__dict__.items() if k.startswith('_')}
        custom_attributes = self._dict_to_str(custom_attributes)
        msg = logging.Formatter.format(self, record)
        msg = '{} ({})'.format(msg, custom_attributes)
        return msg

    def _dict_to_str(self, dictionary):
        result = ['{}={}'.format(k[1:], str(v)) for k, v in dictionary.items()]
        result = ','.join(result)
        return result