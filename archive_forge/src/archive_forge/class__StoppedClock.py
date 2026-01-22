from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class _StoppedClock(object):

    def __init__(self, t):
        self.t = t

    def utcnow(self):
        return self.t