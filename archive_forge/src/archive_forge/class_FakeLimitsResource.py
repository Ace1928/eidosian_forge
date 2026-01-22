import sys
from unittest import mock
from keystoneauth1 import fixture
class FakeLimitsResource(FakeResource):

    class AbsoluteLimit:

        def __init__(self, name, value):
            self.name = name
            self.value = value

    class RateLimit:

        def __init__(self, verb, uri, regex, value, remaining, unit, next_available):
            self.verb = verb
            self.uri = uri
            self.regex = regex
            self.value = value
            self.remaining = remaining
            self.unit = unit
            self.next_available = next_available

    @property
    def absolute(self):
        return [self.AbsoluteLimit(key, value) for key, value in self._info['absolute_limit'].items()]

    @property
    def rate(self):
        rate = self._info['rate_limit']
        return [self.RateLimit(rate['verb'], rate['uri'], rate['regex'], rate['value'], rate['remaining'], rate['unit'], rate['next-available'])]