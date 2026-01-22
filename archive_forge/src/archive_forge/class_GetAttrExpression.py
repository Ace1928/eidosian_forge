from ..utils import SchemaBase
class GetAttrExpression(Expression):

    def __init__(self, group, name):
        super(GetAttrExpression, self).__init__(group=group, name=name)

    def __repr__(self):
        return '{}.{}'.format(self.group, self.name)