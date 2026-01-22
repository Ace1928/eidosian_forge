from boto.dynamodb.types import dynamize_value
class ConditionNoArgs(Condition):
    """
    Abstract class for Conditions that require no arguments, such
    as NULL or NOT_NULL.
    """

    def __repr__(self):
        return '%s' % self.__class__.__name__

    def to_dict(self):
        return {'ComparisonOperator': self.__class__.__name__}