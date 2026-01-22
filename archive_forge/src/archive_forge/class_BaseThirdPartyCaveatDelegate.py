from abc import ABCMeta, abstractmethod
class BaseThirdPartyCaveatDelegate(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(BaseThirdPartyCaveatDelegate, self).__init__(*args, **kwargs)

    @abstractmethod
    def add_third_party_caveat(self, macaroon, location, key, key_id, **kwargs):
        pass