from boto.dynamodb.types import dynamize_value
class ConditionSeveralArgs(Condition):
    """
    Abstract class for conditions that require several argument (ex: IN).
    """

    def __init__(self, values):
        self.values = values

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(self.values))

    def to_dict(self):
        return {'AttributeValueList': [dynamize_value(v) for v in self.values], 'ComparisonOperator': self.__class__.__name__}