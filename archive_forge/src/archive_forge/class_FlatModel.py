from keystone.limit.models import base
class FlatModel(base.ModelBase):
    NAME = 'flat'
    DESCRIPTION = 'Limit enforcement and validation does not take project hierarchy into consideration.'
    MAX_PROJECT_TREE_DEPTH = None

    def check_limit(self, limits):
        return