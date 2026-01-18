from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def utcnow(self):
    return self.t