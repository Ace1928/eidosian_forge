import py
import sys
class KeywordMapper:

    def __init__(self):
        self.keywords2consumer = {}

    def getstate(self):
        return self.keywords2consumer.copy()

    def setstate(self, state):
        self.keywords2consumer.clear()
        self.keywords2consumer.update(state)

    def getconsumer(self, keywords):
        """ return a consumer matching the given keywords.

            tries to find the most suitable consumer by walking, starting from
            the back, the list of keywords, the first consumer matching a
            keyword is returned (falling back to py.log.default)
        """
        for i in range(len(keywords), 0, -1):
            try:
                return self.keywords2consumer[keywords[:i]]
            except KeyError:
                continue
        return self.keywords2consumer.get('default', default_consumer)

    def setconsumer(self, keywords, consumer):
        """ set a consumer for a set of keywords. """
        if isinstance(keywords, str):
            keywords = tuple(filter(None, keywords.split()))
        elif hasattr(keywords, '_keywords'):
            keywords = keywords._keywords
        elif not isinstance(keywords, tuple):
            raise TypeError('key %r is not a string or tuple' % (keywords,))
        if consumer is not None and (not py.builtin.callable(consumer)):
            if not hasattr(consumer, 'write'):
                raise TypeError('%r should be None, callable or file-like' % (consumer,))
            consumer = File(consumer)
        self.keywords2consumer[keywords] = consumer