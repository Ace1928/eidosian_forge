import typing as t
class MapType(DataType):

    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool=True):
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def __repr__(self) -> str:
        return f'MapType({self.keyType}, {self.valueType}, {str(self.valueContainsNull)})'

    def simpleString(self) -> str:
        return f'map<{self.keyType.simpleString()}, {self.valueType.simpleString()}>'

    def jsonValue(self) -> t.Dict[str, t.Any]:
        return {'type': self.typeName(), 'keyType': self.keyType.jsonValue(), 'valueType': self.valueType.jsonValue(), 'valueContainsNull': self.valueContainsNull}