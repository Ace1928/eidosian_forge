from typing import List
from redis import DataError
class NumericField(Field):
    """
    NumericField is used to define a numeric field in a schema definition
    """

    def __init__(self, name: str, **kwargs):
        Field.__init__(self, name, args=[Field.NUMERIC], **kwargs)