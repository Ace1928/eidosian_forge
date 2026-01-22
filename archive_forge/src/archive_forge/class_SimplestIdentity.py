from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class SimplestIdentity(bakery.Identity):

    def __init__(self, user):
        self._identity = user

    def domain(self):
        return ''

    def id(self):
        return self._identity