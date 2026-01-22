from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class ThirdPartyCheckerWithCaveats(bakery.ThirdPartyCaveatChecker):

    def __init__(self, cavs=None):
        if cavs is None:
            cavs = []
        self.cavs = cavs

    def check_third_party_caveat(self, ctx, cav_info):
        return self.cavs