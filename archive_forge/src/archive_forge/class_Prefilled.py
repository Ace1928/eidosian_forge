from collections import defaultdict
import pytest
from modin.config import Parameter
class Prefilled(Parameter, type=vartype):

    @classmethod
    def _get_raw_from_config(cls):
        return varinit