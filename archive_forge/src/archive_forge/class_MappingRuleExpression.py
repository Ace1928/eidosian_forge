import sys
import yaql
from yaql.language import exceptions
from yaql.language import utils
class MappingRuleExpression(Expression):

    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.uses_receiver = False

    def __str__(self):
        return u'{0} => {1}'.format(self.source, self.destination)

    def __call__(self, receiver, context, engine):
        return utils.MappingRule(self.source(receiver, context, engine), self.destination(receiver, context, engine))