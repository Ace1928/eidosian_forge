import ipaddress
from functools import lru_cache
class BaseTzLoader(TimestamptzLoader):
    """
        Load a PostgreSQL timestamptz using the a specific timezone.
        The timezone can be None too, in which case it will be chopped.
        """
    timezone = None

    def load(self, data):
        res = super().load(data)
        return res.replace(tzinfo=self.timezone)