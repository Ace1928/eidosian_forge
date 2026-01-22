from .. import errors, registry
class UnexpectedInventoryFormat(BadInventoryFormat):
    _fmt = 'The inventory was not in the expected format:\n %(msg)s'

    def __init__(self, msg):
        BadInventoryFormat.__init__(self, msg=msg)