import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class PlanPrice:
    """
    Helper class to construct plan price in different zones

    :param  zone_prices: List of prices in different zones in UpCloud
    :type   zone_prices: ```list```

    """

    def __init__(self, zone_prices):
        self._zone_prices = zone_prices

    def get_price(self, plan_name, location=None):
        """
        Returns the plan's price in location. If location
        is not provided returns None

        :param  plan_name: Name of the plan
        :type   plan_name: ```str```

        :param  location: Location, which price is returned (optional)
        :type   location: :class:`.NodeLocation`


        rtype: ``float``
        """
        if location is None:
            return None
        server_plan_name = 'server_plan_' + plan_name
        for zone_price in self._zone_prices:
            if zone_price['name'] == location.id:
                return zone_price.get(server_plan_name, {}).get('price')
        return None