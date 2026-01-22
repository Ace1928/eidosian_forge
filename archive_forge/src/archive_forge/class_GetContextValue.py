import sys
import yaql
from yaql.language import exceptions
from yaql.language import utils
class GetContextValue(Function):

    def __init__(self, path):
        super(GetContextValue, self).__init__('#get_context_data', path)
        self.path = path
        self.uses_receiver = False

    def __str__(self):
        return self.path.value