from typing import List
from redis import DataError
class GeoField(Field):
    """
    GeoField is used to define a geo-indexing field in a schema definition
    """

    def __init__(self, name: str, **kwargs):
        Field.__init__(self, name, args=[Field.GEO], **kwargs)