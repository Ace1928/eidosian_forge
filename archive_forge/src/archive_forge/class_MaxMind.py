import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class MaxMind(Record):
    """Contains data related to your MaxMind account.

    Attributes:

    .. attribute:: queries_remaining

      The number of remaining queries you have
      for the end point you are calling.

      :type: int

    """
    queries_remaining: Optional[int]

    def __init__(self, queries_remaining: Optional[int]=None, **_) -> None:
        self.queries_remaining = queries_remaining